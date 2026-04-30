from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from matcher_agent.features.feature_builder import (
    build_pair_features,
    build_track_audio_lookup,
    build_track_meta_lookup,
)
from matcher_agent.features.playlist_profiles import (
    build_playlist_text_strings,
    build_profiles,
    build_track_text_strings,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from matcher_agent.artifacts.io import save_bundle
from matcher_agent.embeddings import TextEmbedder
from matcher_agent.features.feature_builder import select_model_features
from matcher_agent.training.dataset import build_training_bundle
from matcher_agent.training.metrics import grouped_ranking_metrics

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None
from sklearn.ensemble import GradientBoostingClassifier


@dataclass
class RankerTrainResult:
    auc_pr: float
    auc_roc: float
    feature_columns: list[str]
    rows: int
    ranking_metrics: dict[str, float]


def _build_model(random_state: int):
    if LGBMClassifier is not None:
        base = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.04,
            max_depth=-1,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=0.1,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    else:
        base = GradientBoostingClassifier(random_state=random_state)
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def train_ranker(
    matches_df: pd.DataFrame,
    tracks_df: pd.DataFrame,
    playlists_df: pd.DataFrame,
    *,
    text_embedder: TextEmbedder,
    output_dir: Path,
    model_dir: Path,
    random_state: int = 42,
    semantic_blend: float = 0.5,
    test_size: float = 0.2,
    full_catalog_eval: bool = True,
    negative_sample_ratio: float = 3.0,
    negative_conflict_fraction: float = 0.5,
) -> RankerTrainResult:
    """Train the ranker with a leakage-free pipeline:

    1. Group-split tracks first.
    2. Build playlist profiles (centroids, acceptance_rate, etc.) from
       training matches ONLY -- so test labels never leak into features.
    3. Recompute pair features for both folds against the train-only profiles.
    4. Evaluate against the full playlist catalog when ``full_catalog_eval``
       is True so reported Hit@K/Precision@K reflect deployed behavior.
    """
    matches_df = matches_df.copy()
    matches_df["track_id"] = matches_df["track_id"].astype("string")
    matches_df["playlist_id"] = matches_df["playlist_id"].astype("string")
    matches_df = matches_df.dropna(subset=["track_id", "playlist_id", "label"])

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(matches_df, groups=matches_df["track_id"]))
    train_matches = matches_df.iloc[train_idx].reset_index(drop=True)
    test_matches = matches_df.iloc[test_idx].reset_index(drop=True)
    print(
        f"[Train] Train matches={len(train_matches)} | Test matches={len(test_matches)} | "
        f"Train tracks={train_matches['track_id'].nunique()} | Test tracks={test_matches['track_id'].nunique()}"
    )

    train_bundle = build_training_bundle(
        train_matches,
        tracks_df,
        playlists_df,
        text_embedder=text_embedder,
        semantic_blend=semantic_blend,
    )
    train_df = train_bundle.pair_features
    print(f"[Train] Train pair rows={len(train_df)} positives={int(train_df['label'].sum())}")

    if negative_sample_ratio and negative_sample_ratio > 0:
        train_df = _augment_with_random_negatives(
            train_df,
            train_matches=train_matches,
            playlists_df=playlists_df,
            train_bundle=train_bundle,
            ratio=negative_sample_ratio,
            conflict_fraction=negative_conflict_fraction,
            random_state=random_state,
        )
        print(
            f"[Train] After random-negative augmentation: rows={len(train_df)} "
            f"positives={int(train_df['label'].sum())}"
        )

    test_df = build_pair_features(
        test_matches[["track_id", "playlist_id", "label"]],
        profile_bundle=train_bundle.profile_bundle,
        track_text_emb_by_id=train_bundle.track_text_emb_by_id,
        track_audio_by_id=train_bundle.track_audio_by_id,
        track_meta_by_id=train_bundle.track_meta_by_id,
        track_popularity_by_id=train_bundle.track_popularity_by_id,
    )
    print(f"[Train] Test pair rows={len(test_df)} positives={int(test_df['label'].sum())}")

    feature_cols = select_model_features(train_df)
    print(f"[Train] Using {len(feature_cols)} model features: {feature_cols}")

    X_train = train_df[feature_cols]
    y_train = train_df["label"].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df["label"].astype(int)

    numeric_transform = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    preproc = ColumnTransformer([("num", numeric_transform, feature_cols)], remainder="drop")
    model = _build_model(random_state)
    pipeline = Pipeline([("preproc", preproc), ("model", model)])
    print("[Train] Fitting pipeline.")
    pipeline.fit(X_train, y_train)

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    auc_pr = float(average_precision_score(y_test, y_pred_proba))
    auc_roc = float(roc_auc_score(y_test, y_pred_proba))
    print(f"[Train] auc_pr={auc_pr:.4f} auc_roc={auc_roc:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_df = test_df[["track_id", "playlist_id", "label"]].copy()
    eval_df["pred_proba"] = y_pred_proba
    eval_df.to_csv(output_dir / "ranker_eval.csv", index=False)

    ranking_dict: dict[str, float] = {}
    if full_catalog_eval:
        catalog_eval = _full_catalog_eval(
            test_matches=test_matches,
            playlists_df=playlists_df,
            pipeline=pipeline,
            feature_cols=feature_cols,
            train_bundle=train_bundle,
        )
        catalog_eval.to_csv(output_dir / "full_catalog_eval.csv", index=False)
        catalog_metrics = grouped_ranking_metrics(catalog_eval)
        ranking_dict = catalog_metrics.as_flat_dict()
        # Add genre-relevance which is more meaningful given positive-only bias.
        ranking_dict.update(
            _genre_relevance_eval(catalog_eval, train_bundle, test_matches=test_matches)
        )
        print(
            "[Train] Full-catalog ranking metrics (per held-out track, scored against ALL "
            f"{playlists_df['playlist_id'].nunique()} playlists): "
            + " | ".join(f"{k}={v:.4f}" for k, v in ranking_dict.items())
        )
    else:
        ranking = grouped_ranking_metrics(eval_df)
        ranking_dict = ranking.as_flat_dict()
        print(
            "[Train] Candidate-pool ranking metrics (per held-out track, "
            "scored against historical pitches only): "
            + " | ".join(f"{k}={v:.4f}" for k, v in ranking_dict.items())
        )
    pd.DataFrame([ranking_dict]).to_csv(output_dir / "ranking_metrics.csv", index=False)

    lift_df = eval_df.groupby("playlist_id", as_index=False).agg(
        observed_acceptance=("label", "mean"),
        predicted_acceptance=("pred_proba", "mean"),
        sample_count=("label", "count"),
    )
    lift_df["lift"] = np.where(
        lift_df["observed_acceptance"] > 0,
        lift_df["predicted_acceptance"] / lift_df["observed_acceptance"],
        np.nan,
    )
    lift_df.to_csv(output_dir / "per_playlist_lift.csv", index=False)

    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10, strategy="quantile")
    pd.DataFrame({"prob_true": prob_true, "prob_pred": prob_pred}).to_csv(
        output_dir / "calibration_curve.csv", index=False
    )

    importances = _extract_feature_importances(pipeline, feature_cols, X_test, y_test, random_state)
    feat_imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    feat_imp_df.sort_values("importance", ascending=False).to_csv(
        output_dir / "feature_importance.csv", index=False
    )
    print(
        "[Train] Top features: "
        + ", ".join(
            f"{r.feature}={r.importance:.3f}"
            for r in feat_imp_df.sort_values("importance", ascending=False).head(8).itertuples()
        )
    )

    bundle_to_save = {
        "model": pipeline,
        "feature_columns": feature_cols,
        "metrics": {
            "auc_pr": auc_pr,
            "auc_roc": auc_roc,
            **ranking_dict,
        },
        "config": {
            "semantic_blend": semantic_blend,
            "embedding_model": text_embedder.model_name,
            "audio_feature_cols": train_bundle.profile_bundle.audio_feature_cols,
        },
    }
    save_bundle(bundle_to_save, model_dir)
    print(f"[Train] Model artifacts saved to {model_dir}")

    return RankerTrainResult(
        auc_pr=auc_pr,
        auc_roc=auc_roc,
        feature_columns=feature_cols,
        rows=len(train_df),
        ranking_metrics=ranking_dict,
    )


def _augment_with_random_negatives(
    train_df: pd.DataFrame,
    *,
    train_matches: pd.DataFrame,
    playlists_df: pd.DataFrame,
    train_bundle,
    ratio: float,
    random_state: int,
    conflict_fraction: float = 0.5,
) -> pd.DataFrame:
    """Add `ratio` negative pairs per accepted positive.

    Negatives come in two flavors, in proportions controlled by
    `conflict_fraction`:

    1. **Genre-conflict hard negatives** (playlist-anchored): for each
       positive ``(track, accepted_playlist)`` pair, the accepted playlist's
       canonical tags are used as the genre anchor. We then sample a random
       different playlist whose tags share NOTHING with the anchor. This
       avoids the previous track-side bottleneck where regex tagging of
       short titles like "Bad Habits" produced empty tag sets and the
       sampler had to fall back to random.
    2. **Uniform random negatives**: a random catalog playlist (any genre)
       that the track has not been pitched to. Keeps a baseline of
       "true off-topic" examples so the model doesn't overfit to the
       conflict structure.

    Historical "declines" remain in the positive/negative pool unchanged —
    they're informative as near-miss examples even though genre-controlled.
    """
    from matcher_agent.features.genre_tagger import tag_text

    rng = np.random.default_rng(random_state)
    profiles = train_bundle.profile_bundle.profiles
    all_playlist_ids = playlists_df["playlist_id"].astype(str).drop_duplicates().tolist()

    pitched_by_track: dict[str, set[str]] = {}
    for tid, pid in zip(
        train_matches["track_id"].astype(str), train_matches["playlist_id"].astype(str)
    ):
        pitched_by_track.setdefault(tid, set()).add(pid)

    track_tags_by_id: dict[str, set[str]] = {}
    for tid, meta in train_bundle.track_meta_by_id.items():
        cached = meta.get("_cached_tags")
        if cached is not None:
            track_tags_by_id[tid] = cached
        else:
            text = (
                f"{meta.get('artist','')} {meta.get('track_name','')} "
                f"{meta.get('album','')}"
            ).strip()
            track_tags_by_id[tid] = tag_text(text)

    playlist_tags_by_id: dict[str, set[str]] = {
        pid: prof.tags for pid, prof in profiles.items()
    }
    tagged_playlist_ids = [pid for pid in all_playlist_ids if playlist_tags_by_id.get(pid)]

    positives = train_df[train_df["label"] == 1]
    n_total = int(len(positives) * ratio)
    if n_total <= 0:
        return train_df

    conflict_fraction = max(0.0, min(1.0, conflict_fraction))
    n_conflict_target = int(round(n_total * conflict_fraction))

    positive_pairs = list(
        zip(
            positives["track_id"].astype(str).tolist(),
            positives["playlist_id"].astype(str).tolist(),
        )
    )
    track_ids = [tid for tid, _ in positive_pairs]

    used_neg_pairs: set[tuple[str, str]] = set()
    conflict_records: list[dict] = []
    n_skipped_no_anchor_tags = 0
    n_skipped_no_conflict_found = 0
    if n_conflict_target > 0 and tagged_playlist_ids:
        max_attempts_per_record = 80
        outer_safety_budget = n_conflict_target * 8
        outer_attempts = 0
        while len(conflict_records) < n_conflict_target and outer_attempts < outer_safety_budget:
            outer_attempts += 1
            idx = int(rng.integers(0, len(positive_pairs)))
            tid, accepted_pid = positive_pairs[idx]
            # Prefer the accepted playlist's tags as the genre anchor; fall
            # back to track-text tags only when the playlist itself is
            # untagged. Anchoring on the playlist gives much higher coverage
            # because curator-supplied Xano genres exist for ~75% of the
            # catalog vs ~10-15% of track titles producing usable regex tags.
            anchor_tags = playlist_tags_by_id.get(accepted_pid, set())
            if not anchor_tags:
                anchor_tags = track_tags_by_id.get(tid, set())
            if not anchor_tags:
                n_skipped_no_anchor_tags += 1
                continue

            pitched = pitched_by_track.get(tid, set())
            for _ in range(max_attempts_per_record):
                candidate = tagged_playlist_ids[
                    int(rng.integers(0, len(tagged_playlist_ids)))
                ]
                if candidate in pitched:
                    continue
                if (tid, candidate) in used_neg_pairs:
                    continue
                cand_tags = playlist_tags_by_id.get(candidate, set())
                if not cand_tags:
                    continue
                if anchor_tags & cand_tags:
                    continue
                used_neg_pairs.add((tid, candidate))
                conflict_records.append(
                    {"track_id": tid, "playlist_id": candidate, "label": 0}
                )
                break
            else:
                n_skipped_no_conflict_found += 1

    n_random_target = n_total - len(conflict_records)
    random_records: list[dict] = []
    random_attempts = 0
    random_attempt_budget = max(n_random_target * 20, 1000)
    while len(random_records) < n_random_target and random_attempts < random_attempt_budget:
        random_attempts += 1
        tid = track_ids[int(rng.integers(0, len(track_ids)))]
        pid = all_playlist_ids[int(rng.integers(0, len(all_playlist_ids)))]
        if pid in pitched_by_track.get(tid, set()):
            continue
        if (tid, pid) in used_neg_pairs:
            continue
        used_neg_pairs.add((tid, pid))
        random_records.append({"track_id": tid, "playlist_id": pid, "label": 0})

    print(
        f"[Train] Sampling negatives (playlist-anchored): target={n_total} "
        f"conflict_fraction={conflict_fraction:.2f} "
        f"got_conflict={len(conflict_records)} got_random={len(random_records)} "
        f"skipped_no_anchor_tags={n_skipped_no_anchor_tags} "
        f"skipped_no_conflict_found={n_skipped_no_conflict_found}"
    )

    sampled_records = conflict_records + random_records
    if not sampled_records:
        return train_df
    sampled_df = pd.DataFrame(sampled_records)
    sampled_feats = build_pair_features(
        sampled_df,
        profile_bundle=train_bundle.profile_bundle,
        track_text_emb_by_id=train_bundle.track_text_emb_by_id,
        track_audio_by_id=train_bundle.track_audio_by_id,
        track_meta_by_id=train_bundle.track_meta_by_id,
        track_popularity_by_id=train_bundle.track_popularity_by_id,
    )
    return pd.concat([train_df, sampled_feats], ignore_index=True)


def _genre_relevance_eval(
    catalog_eval: pd.DataFrame,
    train_bundle,
    *,
    test_matches: pd.DataFrame,
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float]:
    """For each test track, what fraction of the top-K predicted playlists
    share at least one genre tag with the track?

    This is a complementary metric to Hit@K. Hit@K is overly strict because
    each test track has only 1-3 historical pitches in the catalog of 1,668
    playlists. Genre relevance captures whether the recommendations are at
    least in the right neighborhood.
    """
    from matcher_agent.features.genre_tagger import tag_text

    # Per-track genre tags (use historical track_name + artist text).
    track_text_by_id: dict[str, str] = {}
    for _, row in test_matches.drop_duplicates(subset=["track_id"]).iterrows():
        tid = str(row["track_id"])
        track_text_by_id[tid] = (
            f"{row.get('artist','')} - {row.get('track_name','')}"
        ).strip()
    track_tags_by_id = {tid: tag_text(t) for tid, t in track_text_by_id.items()}

    profiles = train_bundle.profile_bundle.profiles
    relevance: dict[int, list[float]] = {k: [] for k in ks}
    n_tagged = 0
    for tid, group in catalog_eval.groupby("track_id", sort=False):
        track_tags = track_tags_by_id.get(str(tid), set())
        if not track_tags:
            continue
        n_tagged += 1
        ranked = group.sort_values("pred_proba", ascending=False)
        playlist_ids = ranked["playlist_id"].astype(str).tolist()
        for k in ks:
            top = playlist_ids[:k]
            hits = 0
            for pid in top:
                prof = profiles.get(pid)
                if prof is not None and (track_tags & prof.tags):
                    hits += 1
            relevance[k].append(hits / k)
    out: dict[str, float] = {"genre_eval_groups": float(n_tagged)}
    for k, vals in relevance.items():
        out[f"genre_precision_at_{k}"] = float(np.mean(vals)) if vals else 0.0
    return out


def _full_catalog_eval(
    *,
    test_matches: pd.DataFrame,
    playlists_df: pd.DataFrame,
    pipeline: Pipeline,
    feature_cols: list[str],
    train_bundle,
) -> pd.DataFrame:
    """Score every held-out track against every playlist in the catalog.

    Labels are 1 if the (track, playlist) pair is an accepted historical
    match, 0 otherwise (declined or never-seen). This is the deployment-
    realistic evaluation: the agent must pick the best playlists from the
    full catalog.
    """
    print("[Train] Building full-catalog evaluation table.")
    test_track_ids = test_matches["track_id"].astype("string").drop_duplicates().tolist()
    playlist_ids = playlists_df["playlist_id"].astype("string").drop_duplicates().tolist()
    print(
        f"[Train] Catalog eval: {len(test_track_ids)} test tracks x "
        f"{len(playlist_ids)} playlists = {len(test_track_ids) * len(playlist_ids)} pairs"
    )

    accepted_lookup: set[tuple[str, str]] = set()
    accepted_test = test_matches[test_matches["label"] == 1]
    for _, row in accepted_test.iterrows():
        accepted_lookup.add((str(row["track_id"]), str(row["playlist_id"])))

    pairs = pd.DataFrame(
        [
            {"track_id": tid, "playlist_id": pid}
            for tid in test_track_ids
            for pid in playlist_ids
        ]
    )
    pairs["label"] = [
        1 if (str(t), str(p)) in accepted_lookup else 0
        for t, p in zip(pairs["track_id"], pairs["playlist_id"])
    ]

    feats = build_pair_features(
        pairs,
        profile_bundle=train_bundle.profile_bundle,
        track_text_emb_by_id=train_bundle.track_text_emb_by_id,
        track_audio_by_id=train_bundle.track_audio_by_id,
        track_meta_by_id=train_bundle.track_meta_by_id,
        track_popularity_by_id=train_bundle.track_popularity_by_id,
    )
    probs = pipeline.predict_proba(feats[feature_cols])[:, 1]
    feats["pred_proba"] = probs
    return feats[["track_id", "playlist_id", "label", "pred_proba"]]


def _extract_feature_importances(
    pipeline: Pipeline,
    feature_cols: list[str],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
) -> np.ndarray:
    fitted_model = pipeline.named_steps["model"]
    fold_importances: list[np.ndarray] = []
    calibrated_models = getattr(fitted_model, "calibrated_classifiers_", None)
    if calibrated_models:
        for calibrated in calibrated_models:
            base_estimator = getattr(calibrated, "estimator", None)
            if base_estimator is not None and hasattr(base_estimator, "feature_importances_"):
                vals = np.asarray(base_estimator.feature_importances_, dtype=float)
                if vals.shape[0] == len(feature_cols):
                    fold_importances.append(vals)
        if fold_importances:
            return np.mean(np.vstack(fold_importances), axis=0)
    print("[Train] Falling back to permutation importance.")
    perm = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=5,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=1,
    )
    return np.asarray(perm.importances_mean, dtype=float)
