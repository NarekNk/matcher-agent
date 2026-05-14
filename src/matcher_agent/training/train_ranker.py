from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
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
from matcher_agent.training.metrics import (
    CalibrationResult,
    SliceResult,
    compute_calibration,
    grouped_ranking_metrics,
    median_split_column,
    slice_eval,
)

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None
from sklearn.ensemble import GradientBoostingClassifier


@dataclass
class NegativeSamplingConfig:
    """Controls the three-tier negative-sampling strategy.

    Negatives are split into three pools whose fractions must sum to ≤ 1.0:

    1. **Genre-conflict** (``conflict_fraction``): playlists with zero
       tag overlap against the accepted playlist's tags.
    2. **Near-miss** (``near_miss_fraction``): playlists with high
       semantic similarity to the accepted playlist but not the accepted
       one.  These are the hardest negatives — they look like the right
       answer on genre/embedding but weren't the actual match.
    3. **Random** (remainder): uniform catalog samples (optionally
       stratified by tier/popularity so all bands are represented).
    """

    ratio: float = 5.0
    conflict_fraction: float = 0.33
    near_miss_fraction: float = 0.33
    popularity_stratified: bool = True
    near_miss_top_k: int = 20

    def __post_init__(self) -> None:
        total = self.conflict_fraction + self.near_miss_fraction
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"conflict_fraction + near_miss_fraction must be <= 1.0, "
                f"got {total:.3f}"
            )
        self.conflict_fraction = max(0.0, min(1.0, self.conflict_fraction))
        self.near_miss_fraction = max(0.0, min(1.0, self.near_miss_fraction))

    @property
    def random_fraction(self) -> float:
        return max(0.0, 1.0 - self.conflict_fraction - self.near_miss_fraction)

    @classmethod
    def from_legacy(
        cls, ratio: float, conflict_fraction: float
    ) -> NegativeSamplingConfig:
        """Build from the old two-parameter interface (backward compat)."""
        return cls(
            ratio=ratio,
            conflict_fraction=conflict_fraction,
            near_miss_fraction=0.0,
            popularity_stratified=False,
        )


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
    sampling_config: NegativeSamplingConfig | None = None,
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

    neg_cfg = sampling_config or NegativeSamplingConfig.from_legacy(
        negative_sample_ratio, negative_conflict_fraction
    )
    if neg_cfg.ratio > 0:
        train_df = _augment_with_negatives(
            train_df,
            train_matches=train_matches,
            playlists_df=playlists_df,
            train_bundle=train_bundle,
            config=neg_cfg,
            random_state=random_state,
        )
        print(
            f"[Train] After negative augmentation: rows={len(train_df)} "
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

    # --- Calibration analysis ---
    calibration = compute_calibration(y_test.to_numpy(), y_pred_proba)
    print(f"[Train] ECE={calibration.ece:.4f} (n_bins={calibration.n_bins})")
    pd.DataFrame({
        "bin_edge_lo": calibration.bin_edges[:-1],
        "bin_edge_hi": calibration.bin_edges[1:],
        "prob_true": calibration.bin_true_freq,
        "prob_pred": calibration.bin_pred_mean,
        "count": calibration.bin_counts,
    }).to_csv(output_dir / "calibration_curve.csv", index=False)

    # --- Full-catalog ranking evaluation ---
    ranking_dict: dict[str, float] = {}
    slice_results: list[dict] = []
    genre_metrics: dict[str, float] = {}
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
        genre_metrics = _genre_relevance_eval(
            catalog_eval, train_bundle, test_matches=test_matches,
        )
        ranking_dict.update(genre_metrics)
        print(
            "[Train] Full-catalog ranking metrics (per held-out track, scored against ALL "
            f"{playlists_df['playlist_id'].nunique()} playlists): "
            + " | ".join(f"{k}={v:.4f}" for k, v in ranking_dict.items())
        )

        # --- Slice analysis ---
        slice_results = _run_slice_analysis(
            catalog_eval, train_bundle=train_bundle,
        )
        if slice_results:
            pd.DataFrame(slice_results).to_csv(
                output_dir / "slice_analysis.csv", index=False,
            )
            print(f"[Train] Slice analysis: {len(slice_results)} slices written.")
    else:
        ranking = grouped_ranking_metrics(eval_df)
        ranking_dict = ranking.as_flat_dict()
        print(
            "[Train] Candidate-pool ranking metrics (per held-out track, "
            "scored against historical pitches only): "
            + " | ".join(f"{k}={v:.4f}" for k, v in ranking_dict.items())
        )
    pd.DataFrame([ranking_dict]).to_csv(output_dir / "ranking_metrics.csv", index=False)

    # --- Per-playlist lift ---
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

    # --- Feature importance ---
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

    # --- Evaluation report (JSON) ---
    eval_report = _build_eval_report(
        auc_pr=auc_pr,
        auc_roc=auc_roc,
        ranking_dict=ranking_dict,
        genre_metrics=genre_metrics,
        calibration=calibration,
        slice_results=slice_results,
        feature_importances=feat_imp_df.sort_values("importance", ascending=False)
        .head(20)
        .to_dict(orient="records"),
        n_train=len(train_df),
        n_test=len(test_df),
        n_features=len(feature_cols),
        feature_cols=feature_cols,
    )
    import json
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(eval_report, indent=2, default=str)
    )
    print(f"[Train] Evaluation report saved to {output_dir / 'evaluation_report.json'}")

    # --- Save model artifacts ---
    bundle_to_save = {
        "model": pipeline,
        "feature_columns": feature_cols,
        "metrics": {
            "auc_pr": auc_pr,
            "auc_roc": auc_roc,
            "ece": calibration.ece,
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
    """Legacy wrapper — delegates to :func:`_augment_with_negatives`."""
    config = NegativeSamplingConfig.from_legacy(ratio, conflict_fraction)
    return _augment_with_negatives(
        train_df,
        train_matches=train_matches,
        playlists_df=playlists_df,
        train_bundle=train_bundle,
        config=config,
        random_state=random_state,
    )


def _cosine_vec(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _precompute_similar_playlists(
    playlist_text_emb_by_id: dict[str, np.ndarray],
    k: int = 20,
) -> dict[str, list[str]]:
    """For each playlist, precompute top-K most similar playlists by cosine."""
    pids = list(playlist_text_emb_by_id.keys())
    if len(pids) < 2:
        return {}
    emb_matrix = np.vstack(
        [playlist_text_emb_by_id[pid].astype(np.float64) for pid in pids]
    )
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    emb_normed = emb_matrix / norms
    sim_matrix = emb_normed @ emb_normed.T

    k = min(k, len(pids) - 1)
    similar: dict[str, list[str]] = {}
    for i, pid in enumerate(pids):
        sims = sim_matrix[i].copy()
        sims[i] = -2.0
        top_indices = np.argsort(sims)[-k:][::-1]
        similar[pid] = [pids[j] for j in top_indices if sims[j] > -1.0]
    return similar


def _build_tier_buckets(
    all_playlist_ids: list[str],
    profiles: dict,
) -> list[list[str]]:
    """Bucket playlists by tier (1-4) for stratified sampling.

    Playlists with no tier go into a catch-all bucket. Returns a list of
    non-empty buckets; downstream sampling picks a bucket uniformly then
    picks a playlist within it.
    """
    buckets: dict[int, list[str]] = {}
    no_tier: list[str] = []
    for pid in all_playlist_ids:
        prof = profiles.get(pid)
        if prof is not None and prof.tier is not None:
            buckets.setdefault(prof.tier, []).append(pid)
        else:
            no_tier.append(pid)
    out: list[list[str]] = [v for v in buckets.values() if v]
    if no_tier:
        out.append(no_tier)
    return out if out else [all_playlist_ids]


def _augment_with_negatives(
    train_df: pd.DataFrame,
    *,
    train_matches: pd.DataFrame,
    playlists_df: pd.DataFrame,
    train_bundle,
    config: NegativeSamplingConfig,
    random_state: int,
) -> pd.DataFrame:
    """Three-tier negative sampling: conflict + near-miss + random.

    1. **Genre-conflict** hard negatives: playlists with zero tag overlap
       against the accepted playlist's tags.
    2. **Near-miss** hard negatives: playlists semantically similar to the
       accepted one (high cosine similarity of text embeddings) but not
       the accepted playlist.  These teach the model fine-grained
       discrimination between similar playlists.
    3. **Random** negatives: catalog playlists the track was never
       pitched to.  When ``config.popularity_stratified`` is True,
       sampling is stratified by playlist tier so all popularity bands
       are represented equally.

    Shortfall from any tier rolls into the random pool.
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
    n_total = int(len(positives) * config.ratio)
    if n_total <= 0:
        return train_df

    n_conflict_target = int(round(n_total * config.conflict_fraction))
    n_near_miss_target = int(round(n_total * config.near_miss_fraction))

    positive_pairs = list(
        zip(
            positives["track_id"].astype(str).tolist(),
            positives["playlist_id"].astype(str).tolist(),
        )
    )
    track_ids = [tid for tid, _ in positive_pairs]
    used_neg_pairs: set[tuple[str, str]] = set()

    # ---- Tier 1: Genre-conflict negatives ----
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

    # ---- Tier 2: Near-miss negatives ----
    near_miss_records: list[dict] = []
    if n_near_miss_target > 0 and len(all_playlist_ids) >= 2:
        similar_playlists = _precompute_similar_playlists(
            train_bundle.playlist_text_emb_by_id, k=config.near_miss_top_k,
        )
        nm_safety_budget = n_near_miss_target * 8
        nm_attempts = 0
        while len(near_miss_records) < n_near_miss_target and nm_attempts < nm_safety_budget:
            nm_attempts += 1
            idx = int(rng.integers(0, len(positive_pairs)))
            tid, accepted_pid = positive_pairs[idx]
            pitched = pitched_by_track.get(tid, set())
            neighbors = similar_playlists.get(accepted_pid, [])
            if not neighbors:
                continue
            # Shuffle order of neighbors for this attempt so we don't always
            # pick the single most similar playlist.
            cand_idx = int(rng.integers(0, min(len(neighbors), config.near_miss_top_k)))
            candidate = neighbors[cand_idx]
            if candidate in pitched:
                continue
            if (tid, candidate) in used_neg_pairs:
                continue
            used_neg_pairs.add((tid, candidate))
            near_miss_records.append(
                {"track_id": tid, "playlist_id": candidate, "label": 0}
            )

    # ---- Tier 3: Random negatives (optionally stratified by tier) ----
    n_random_target = n_total - len(conflict_records) - len(near_miss_records)
    random_records: list[dict] = []
    if n_random_target > 0:
        if config.popularity_stratified:
            tier_buckets = _build_tier_buckets(all_playlist_ids, profiles)
        else:
            tier_buckets = [all_playlist_ids]

        random_attempts = 0
        random_attempt_budget = max(n_random_target * 20, 1000)
        while len(random_records) < n_random_target and random_attempts < random_attempt_budget:
            random_attempts += 1
            tid = track_ids[int(rng.integers(0, len(track_ids)))]
            bucket = tier_buckets[int(rng.integers(0, len(tier_buckets)))]
            pid = bucket[int(rng.integers(0, len(bucket)))]
            if pid in pitched_by_track.get(tid, set()):
                continue
            if (tid, pid) in used_neg_pairs:
                continue
            used_neg_pairs.add((tid, pid))
            random_records.append({"track_id": tid, "playlist_id": pid, "label": 0})

    print(
        f"[Train] Negative sampling: target={n_total} "
        f"conflict={config.conflict_fraction:.0%} "
        f"near_miss={config.near_miss_fraction:.0%} "
        f"random={config.random_fraction:.0%} "
        f"stratified={config.popularity_stratified} | "
        f"got_conflict={len(conflict_records)} "
        f"got_near_miss={len(near_miss_records)} "
        f"got_random={len(random_records)} "
        f"skipped_no_anchor_tags={n_skipped_no_anchor_tags} "
        f"skipped_no_conflict_found={n_skipped_no_conflict_found}"
    )

    sampled_records = conflict_records + near_miss_records + random_records
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
    """Genre-aware ranking quality for each test track.

    Computes three complementary genre metrics for each K:

    1. ``genre_precision_at_K`` (all tags): fraction of top-K playlists
       sharing *any* genre tag with the track.
    2. ``genre_primary_precision_at_K``: fraction of top-K playlists
       whose *primary* Xano genres overlap with the track's tags.
       This is stricter — a Rock playlist with subgenre 'Blues Rock' is
       not counted as a match for a Blues track here.
    3. ``genre_conflict_rate_at_K``: fraction of top-K playlists that
       have a genre *conflict* with the track (using the same
       ``has_conflict`` logic as training).
    """
    from matcher_agent.features.genre_tagger import has_conflict, tag_text

    track_text_by_id: dict[str, str] = {}
    for _, row in test_matches.drop_duplicates(subset=["track_id"]).iterrows():
        tid = str(row["track_id"])
        track_text_by_id[tid] = (
            f"{row.get('artist','')} - {row.get('track_name','')}"
        ).strip()
    track_tags_by_id = {tid: tag_text(t) for tid, t in track_text_by_id.items()}

    profiles = train_bundle.profile_bundle.profiles
    all_tag_prec: dict[int, list[float]] = {k: [] for k in ks}
    primary_prec: dict[int, list[float]] = {k: [] for k in ks}
    conflict_rate: dict[int, list[float]] = {k: [] for k in ks}
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
            all_hits = 0
            primary_hits = 0
            conflicts = 0
            for pid in top:
                prof = profiles.get(pid)
                if prof is None:
                    continue
                if track_tags & prof.tags:
                    all_hits += 1
                if track_tags & prof.primary_tags:
                    primary_hits += 1
                if has_conflict(track_tags, prof.tags):
                    conflicts += 1
            all_tag_prec[k].append(all_hits / k)
            primary_prec[k].append(primary_hits / k)
            conflict_rate[k].append(conflicts / k)

    out: dict[str, float] = {"genre_eval_groups": float(n_tagged)}
    for k in ks:
        out[f"genre_precision_at_{k}"] = (
            float(np.mean(all_tag_prec[k])) if all_tag_prec[k] else 0.0
        )
        out[f"genre_primary_precision_at_{k}"] = (
            float(np.mean(primary_prec[k])) if primary_prec[k] else 0.0
        )
        out[f"genre_conflict_rate_at_{k}"] = (
            float(np.mean(conflict_rate[k])) if conflict_rate[k] else 0.0
        )
    return out


_CATALOG_SLICE_COLS = [
    "track_audio_available",
    "track_popularity_norm",
    "popularity_available",
]


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

    Slice-relevant columns (``track_audio_available``,
    ``track_popularity_norm``, ``playlist_size_log``) are preserved so
    callers can break down metrics without recomputing features.
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
    keep_cols = [
        "track_id", "playlist_id", "label", "pred_proba",
        *(c for c in _CATALOG_SLICE_COLS if c in feats.columns),
    ]
    return feats[keep_cols]


def _run_slice_analysis(
    catalog_eval: pd.DataFrame,
    *,
    train_bundle,
) -> list[dict]:
    """Run slice analysis over the full-catalog evaluation DataFrame.

    Slices:
      * ``audio_slice``: tracks WITH vs WITHOUT audio features
      * ``popularity_slice``: high vs low track popularity (median split)
      * ``accepted_count_slice``: playlists with many vs few accepted tracks
      * ``primary_genre``: group by the playlist's first primary genre tag

    Returns a list of flat dicts (one per slice bucket) that can be written
    to CSV and included in the JSON evaluation report.
    """
    profiles = train_bundle.profile_bundle.profiles
    df = catalog_eval.copy()

    # 1. Audio available slice (per-track)
    if "track_audio_available" in df.columns:
        df["audio_slice"] = np.where(
            df["track_audio_available"] >= 1.0, "with_audio", "without_audio",
        )
    else:
        df["audio_slice"] = "unknown"

    # 2. Popularity slice (per-track, median split)
    if "track_popularity_norm" in df.columns and "popularity_available" in df.columns:
        known = df[df["popularity_available"] >= 1.0]
        if not known.empty:
            pop_median = known["track_popularity_norm"].median()
            df["popularity_slice"] = np.where(
                df["track_popularity_norm"] >= pop_median, "high_popularity", "low_popularity",
            )
            df.loc[df["popularity_available"] < 1.0, "popularity_slice"] = "unknown_popularity"
        else:
            df["popularity_slice"] = "unknown_popularity"
    else:
        df["popularity_slice"] = "unknown_popularity"

    # 3. Playlist accepted-count slice (per-playlist, median split)
    pl_accepted = {
        pid: prof.accepted_count for pid, prof in profiles.items()
    }
    if pl_accepted:
        median_acc = float(np.median(list(pl_accepted.values())))
        df["pl_accepted_bucket"] = df["playlist_id"].astype(str).map(
            lambda pid: "many_accepted" if pl_accepted.get(pid, 0) >= median_acc else "few_accepted"
        )
    else:
        df["pl_accepted_bucket"] = "unknown"

    # 4. Primary genre slice (per-playlist)
    def _primary_genre(pid: str) -> str:
        prof = profiles.get(pid)
        if prof is None or not prof.primary_tags:
            return "untagged"
        return sorted(prof.primary_tags)[0]

    df["primary_genre"] = df["playlist_id"].astype(str).map(_primary_genre)

    results: list[dict] = []
    for col in ("audio_slice", "popularity_slice", "pl_accepted_bucket", "primary_genre"):
        slices = slice_eval(df, slice_col=col)
        for s in slices:
            results.append({
                "slice_name": s.slice_name,
                "slice_value": s.slice_value,
                "n_groups": s.n_groups,
                **s.metrics,
            })
    return results


def _build_eval_report(
    *,
    auc_pr: float,
    auc_roc: float,
    ranking_dict: dict[str, float],
    genre_metrics: dict[str, float],
    calibration: CalibrationResult,
    slice_results: list[dict],
    feature_importances: list[dict],
    n_train: int,
    n_test: int,
    n_features: int,
    feature_cols: list[str],
) -> dict:
    """Assemble the comprehensive evaluation report dict written as JSON."""
    return {
        "binary_metrics": {
            "auc_pr": auc_pr,
            "auc_roc": auc_roc,
        },
        "ranking_metrics": ranking_dict,
        "genre_metrics": genre_metrics,
        "calibration": {
            "ece": calibration.ece,
            "n_bins": calibration.n_bins,
            "bins": [
                {
                    "bin_lo": calibration.bin_edges[i],
                    "bin_hi": calibration.bin_edges[i + 1],
                    "true_freq": calibration.bin_true_freq[i],
                    "pred_mean": calibration.bin_pred_mean[i],
                    "count": calibration.bin_counts[i],
                }
                for i in range(calibration.n_bins)
            ],
        },
        "slices": slice_results,
        "feature_importances": feature_importances,
        "dataset": {
            "n_train_rows": n_train,
            "n_test_rows": n_test,
            "n_features": n_features,
            "feature_columns": feature_cols,
        },
    }


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
