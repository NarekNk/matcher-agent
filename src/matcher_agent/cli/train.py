from __future__ import annotations

import argparse

from matcher_agent.config import get_settings
from matcher_agent.data.repository import DataRepository
from matcher_agent.embeddings import TextEmbedder
from matcher_agent.storage.parquet_store import ParquetStore
from matcher_agent.training.train_ranker import NegativeSamplingConfig, train_ranker


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the genre-aware acceptance ranker.")
    parser.add_argument(
        "--tracks-csv",
        default="output/training_data.csv",
        help="Track features CSV path (audio features export).",
    )
    parser.add_argument(
        "--semantic-blend",
        type=float,
        default=None,
        help="Weight on the playlist's own text embedding when blending with the "
        "centroid of accepted tracks. Defaults to SEMANTIC_BLEND env var (0.25).",
    )
    parser.add_argument(
        "--negative-sample-ratio",
        type=float,
        default=None,
        help="Negative pairs sampled per accepted positive. Defaults to "
        "NEGATIVE_SAMPLE_RATIO env var (5.0). Use 0 to disable.",
    )
    parser.add_argument(
        "--negative-conflict-fraction",
        type=float,
        default=None,
        help="Fraction of negatives that are genre-conflicting. "
        "Defaults to NEGATIVE_CONFLICT_FRACTION env var (0.5).",
    )
    parser.add_argument(
        "--near-miss-fraction",
        type=float,
        default=None,
        help="Fraction of negatives that are near-miss (semantically "
        "similar to the accepted playlist). Defaults to "
        "NEGATIVE_NEAR_MISS_FRACTION env var (0.33).",
    )
    parser.add_argument(
        "--no-popularity-stratified",
        action="store_true",
        help="Disable popularity-stratified random negatives "
        "(use uniform random instead).",
    )
    args = parser.parse_args()

    print("[TrainCLI] Starting training run.")
    settings = get_settings()
    semantic_blend = (
        args.semantic_blend if args.semantic_blend is not None else settings.semantic_blend
    )

    sampling_config = NegativeSamplingConfig(
        ratio=(
            args.negative_sample_ratio
            if args.negative_sample_ratio is not None
            else settings.negative_sample_ratio
        ),
        conflict_fraction=(
            args.negative_conflict_fraction
            if args.negative_conflict_fraction is not None
            else settings.negative_conflict_fraction
        ),
        near_miss_fraction=(
            args.near_miss_fraction
            if args.near_miss_fraction is not None
            else settings.negative_near_miss_fraction
        ),
        popularity_stratified=(
            False if args.no_popularity_stratified
            else settings.negative_popularity_stratified
        ),
    )

    repo = DataRepository(ParquetStore(settings.data_dir))
    labeled_matches, rejects = repo.load_labeled_historical_matches()
    print(f"[TrainCLI] Labeled matches={len(labeled_matches)} rejects={len(rejects)}")
    if not rejects.empty:
        rejects.to_csv(settings.output_dir / "label_mapping_rejects.csv", index=False)

    playlists_df = repo.load_playlists()
    tracks_df = repo.load_tracks_from_export(args.tracks_csv)
    print(f"[TrainCLI] Playlists={len(playlists_df)} tracks={len(tracks_df)}")

    embedder = TextEmbedder(
        cache_path=settings.embeddings_dir / "text_embeddings.parquet",
        model_name=settings.text_embedding_model,
        device=settings.text_embedding_device,
    )

    print(
        f"[TrainCLI] Training config: semantic_blend={semantic_blend} "
        f"sampling={sampling_config}"
    )
    result = train_ranker(
        matches_df=labeled_matches,
        tracks_df=tracks_df,
        playlists_df=playlists_df,
        text_embedder=embedder,
        output_dir=settings.output_dir,
        model_dir=settings.model_dir,
        random_state=settings.random_state,
        semantic_blend=semantic_blend,
        sampling_config=sampling_config,
    )
    print(
        f"[TrainCLI] Completed: rows={result.rows} auc_pr={result.auc_pr:.4f} "
        f"auc_roc={result.auc_roc:.4f} ranking={result.ranking_metrics}"
    )


if __name__ == "__main__":
    main()
