"""Grid-search post-model penalty values against full-catalog Hit@K and MRR.

Loads the trained model and test-set from disk, scores every test track
against the full playlist catalog, then applies each penalty combination
on top of the raw model scores and evaluates Hit@K + MRR.  Outputs the
best penalty combination to stdout and a CSV report.

Run:
    PYTHONPATH=src python scripts/optimize_penalties.py

Optional arguments:
    --model-dir      Path to saved model artifacts   (default: artifacts/)
    --tracks-csv     Audio-features export CSV        (default: output/training_data.csv)
    --output         Report CSV path                  (default: output/penalty_grid_results.csv)
    --quick          Coarser grid for faster iteration
"""
from __future__ import annotations

import argparse
import itertools
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from matcher_agent.artifacts.io import load_bundle
from matcher_agent.config import get_settings
from matcher_agent.data.repository import DataRepository
from matcher_agent.embeddings import TextEmbedder
from matcher_agent.features.attribute_normalizer import normalize_attribute_labels
from matcher_agent.features.feature_builder import (
    build_pair_features,
    build_track_audio_lookup,
    build_track_meta_lookup,
    select_model_features,
)
from matcher_agent.features.genre_normalizer import normalize_xano_labels
from matcher_agent.features.genre_tagger import has_conflict, tag_text
from matcher_agent.features.playlist_profiles import (
    build_playlist_text_strings,
    build_profiles,
    build_track_popularity_lookup,
    build_track_text_strings,
)
from matcher_agent.storage.parquet_store import ParquetStore
from matcher_agent.training.dataset import build_training_bundle
from matcher_agent.training.metrics import grouped_ranking_metrics


def _apply_penalties(
    catalog_df: pd.DataFrame,
    *,
    profiles: dict,
    no_match_penalty: float,
    untagged_penalty: float,
    subgenre_only_penalty: float,
    soft_attr_penalty: float,
    language_penalty: float,
    broadtag_threshold: int,
    track_tags_by_id: dict[str, set[str]],
    track_soft_by_id: dict[str, dict[str, set[str]]],
) -> pd.Series:
    """Apply a penalty combination to raw model scores and return adjusted scores."""
    scores = catalog_df["pred_proba"].to_numpy(dtype=np.float64).copy()
    track_ids = catalog_df["track_id"].astype(str).to_numpy()
    playlist_ids = catalog_df["playlist_id"].astype(str).to_numpy()

    for i in range(len(scores)):
        tid = str(track_ids[i])
        pid = str(playlist_ids[i])
        track_tags = track_tags_by_id.get(tid, set())
        prof = profiles.get(pid)

        if prof is None or not track_tags:
            continue

        # --- Explicit genre overlap penalty ---
        if not prof.tags:
            scores[i] *= untagged_penalty
            continue
        primary = prof.primary_tags
        if primary and (track_tags & primary):
            tier_mult = 1.0
        elif track_tags & prof.tags:
            tier_mult = subgenre_only_penalty
        else:
            tier_mult = no_match_penalty

        breadth = len(primary) if primary else len(prof.tags)
        if breadth > broadtag_threshold:
            breadth_mult = math.sqrt(float(broadtag_threshold) / float(breadth))
        else:
            breadth_mult = 1.0
        scores[i] *= tier_mult * breadth_mult

        # --- Soft attribute penalty ---
        track_soft = track_soft_by_id.get(tid, {})
        if track_soft and prof is not None:
            pl_attrs = prof.soft_attribute_sets()
            for attr_name, track_vals in track_soft.items():
                if not track_vals:
                    continue
                pl_vals = pl_attrs.get(attr_name, set())
                if pl_vals and not (track_vals & pl_vals):
                    if attr_name == "languages":
                        scores[i] *= language_penalty
                    else:
                        scores[i] *= soft_attr_penalty

    return pd.Series(scores, index=catalog_df.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-search post-model penalty values.")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--tracks-csv", default="output/training_data.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--quick", action="store_true", help="Coarser grid for faster iteration."
    )
    args = parser.parse_args()

    settings = get_settings()
    model_dir = Path(args.model_dir) if args.model_dir else settings.model_dir
    output_path = Path(args.output) if args.output else settings.output_dir / "penalty_grid_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    repo = DataRepository(ParquetStore(settings.data_dir))
    labeled_matches, _ = repo.load_labeled_historical_matches()
    playlists_df = repo.load_playlists()
    tracks_df = repo.load_tracks_from_export(args.tracks_csv)

    embedder = TextEmbedder(
        cache_path=settings.embeddings_dir / "text_embeddings.parquet",
        model_name=settings.text_embedding_model,
        device=settings.text_embedding_device,
    )

    # Reproduce the train/test split used during training.
    matches_df = labeled_matches.copy()
    matches_df["track_id"] = matches_df["track_id"].astype("string")
    matches_df["playlist_id"] = matches_df["playlist_id"].astype("string")
    matches_df = matches_df.dropna(subset=["track_id", "playlist_id", "label"])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=settings.random_state)
    _, test_idx = next(gss.split(matches_df, groups=matches_df["track_id"]))
    test_matches = matches_df.iloc[test_idx].reset_index(drop=True)
    train_matches = matches_df.iloc[
        sorted(set(range(len(matches_df))) - set(test_idx))
    ].reset_index(drop=True)

    print(f"[Optimize] Test tracks={test_matches['track_id'].nunique()}")

    # Build profiles from train-only data (no leakage).
    train_bundle = build_training_bundle(
        train_matches, tracks_df, playlists_df,
        text_embedder=embedder,
        semantic_blend=settings.semantic_blend,
    )

    # Score every test track against every playlist.
    bundle = load_bundle(model_dir)
    pipeline = bundle["model"]
    feature_cols = bundle["feature_columns"]

    test_track_ids = test_matches["track_id"].astype("string").drop_duplicates().tolist()
    playlist_ids = playlists_df["playlist_id"].astype("string").drop_duplicates().tolist()
    print(
        f"[Optimize] Scoring {len(test_track_ids)} tracks x {len(playlist_ids)} playlists"
    )

    accepted_lookup: set[tuple[str, str]] = set()
    for _, row in test_matches[test_matches["label"] == 1].iterrows():
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
    raw_probs = pipeline.predict_proba(feats[feature_cols])[:, 1]
    catalog_df = feats[["track_id", "playlist_id", "label"]].copy()
    catalog_df["pred_proba"] = raw_probs

    # Build per-track tag sets (using the authoritative approach from service.py).
    track_tags_by_id: dict[str, set[str]] = {}
    for tid, meta in train_bundle.track_meta_by_id.items():
        text = (
            f"{meta.get('artist','')} {meta.get('track_name','')} {meta.get('album','')}"
        ).strip()
        track_tags_by_id[tid] = meta.get("_cached_tags") or tag_text(text)

    # No track-side soft attributes in historical data; empty dict per track.
    track_soft_by_id: dict[str, dict[str, set[str]]] = {}

    profiles = train_bundle.profile_bundle.profiles

    # Baseline (no penalties).
    baseline_metrics = grouped_ranking_metrics(catalog_df)
    baseline = baseline_metrics.as_flat_dict()
    print(
        f"[Optimize] Baseline (no penalties): "
        f"MRR={baseline['mrr']:.4f} "
        f"Hit@1={baseline.get('hit_at_1', 0):.4f} "
        f"Hit@5={baseline.get('hit_at_5', 0):.4f} "
        f"Hit@10={baseline.get('hit_at_10', 0):.4f}"
    )

    # --- Broadtag guard impact analysis ---
    print("\n[Optimize] === Over-tagging guard impact analysis ===")
    for threshold in [4, 6, 8]:
        n_penalized = 0
        for pid, prof in profiles.items():
            primary = prof.primary_tags
            breadth = len(primary) if primary else len(prof.tags)
            if breadth > threshold:
                n_penalized += 1
        print(
            f"  threshold={threshold}: {n_penalized}/{len(profiles)} playlists "
            f"({100*n_penalized/max(len(profiles),1):.1f}%) would be penalized"
        )

    # --- Grid search ---
    if args.quick:
        no_match_grid = [0.02, 0.05, 0.1]
        untagged_grid = [0.2, 0.3, 0.5]
        subgenre_grid = [0.3, 0.5, 0.7]
        soft_attr_grid = [0.5, 0.7, 0.9, 1.0]
        language_grid = [0.2, 0.3, 0.5]
        broadtag_grid = [4, 6, 8]
    else:
        no_match_grid = [0.01, 0.02, 0.05, 0.1, 0.15]
        untagged_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
        subgenre_grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        soft_attr_grid = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        language_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
        broadtag_grid = [4, 5, 6, 7, 8]

    combos = list(itertools.product(
        no_match_grid, untagged_grid, subgenre_grid,
        soft_attr_grid, language_grid, broadtag_grid,
    ))
    print(f"\n[Optimize] Grid search: {len(combos)} combinations")
    start = time.time()

    results: list[dict] = []
    best_mrr = -1.0
    best_combo: dict = {}

    for idx, (nm, ut, sg, sa, lg, bt) in enumerate(combos):
        adjusted = _apply_penalties(
            catalog_df,
            profiles=profiles,
            no_match_penalty=nm,
            untagged_penalty=ut,
            subgenre_only_penalty=sg,
            soft_attr_penalty=sa,
            language_penalty=lg,
            broadtag_threshold=bt,
            track_tags_by_id=track_tags_by_id,
            track_soft_by_id=track_soft_by_id,
        )
        eval_df = catalog_df[["track_id", "playlist_id", "label"]].copy()
        eval_df["pred_proba"] = adjusted
        metrics = grouped_ranking_metrics(eval_df)
        flat = metrics.as_flat_dict()

        row = {
            "no_match_penalty": nm,
            "untagged_penalty": ut,
            "subgenre_only_penalty": sg,
            "soft_attr_penalty": sa,
            "language_penalty": lg,
            "broadtag_threshold": bt,
            **flat,
        }
        results.append(row)

        if flat["mrr"] > best_mrr:
            best_mrr = flat["mrr"]
            best_combo = row

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - start
            print(
                f"  [{idx+1}/{len(combos)}] elapsed={elapsed:.0f}s "
                f"best_mrr={best_mrr:.4f}"
            )

    elapsed = time.time() - start
    print(f"\n[Optimize] Grid search completed in {elapsed:.1f}s")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("mrr", ascending=False)
    results_df.to_csv(output_path, index=False)
    print(f"[Optimize] Results saved to {output_path}")

    print("\n" + "=" * 60)
    print("BEST PENALTY COMBINATION (by MRR):")
    print("=" * 60)
    for k, v in best_combo.items():
        print(f"  {k}: {v}")

    # Also report the top combos by hit@5 for comparison.
    top_hit5 = results_df.sort_values("hit_at_5", ascending=False).head(1).iloc[0]
    print(f"\nBest by Hit@5:")
    for col in ["no_match_penalty", "untagged_penalty", "subgenre_only_penalty",
                "soft_attr_penalty", "language_penalty", "broadtag_threshold",
                "mrr", "hit_at_1", "hit_at_5", "hit_at_10"]:
        if col in top_hit5:
            print(f"  {col}: {top_hit5[col]}")

    # Compare best vs current defaults vs baseline.
    print(f"\nComparison:")
    print(f"  Baseline (no penalties):  MRR={baseline['mrr']:.4f}")
    current_df = results_df[
        (results_df["no_match_penalty"] == 0.02)
        & (results_df["untagged_penalty"] == 0.3)
        & (results_df["subgenre_only_penalty"] == 0.4)
        & (results_df["soft_attr_penalty"] == 0.7)
        & (results_df["language_penalty"] == 0.3)
        & (results_df["broadtag_threshold"] == 4)
    ]
    if not current_df.empty:
        cur = current_df.iloc[0]
        print(f"  Current defaults:         MRR={cur['mrr']:.4f}")
    print(f"  Best found:               MRR={best_combo['mrr']:.4f}")


if __name__ == "__main__":
    main()
