from __future__ import annotations

import argparse

from matcher_agent.config import get_settings
from matcher_agent.data.repository import DataRepository
from matcher_agent.embeddings import TextEmbedder
from matcher_agent.features.playlist_profiles import (
    build_playlist_text_strings,
    build_track_text_strings,
)
from matcher_agent.storage.parquet_store import ParquetStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute text embeddings for playlists and tracks (cached to parquet)."
    )
    parser.add_argument("--tracks-csv", default="output/training_data.csv")
    args = parser.parse_args()

    settings = get_settings()
    embedder = TextEmbedder(
        cache_path=settings.embeddings_dir / "text_embeddings.parquet",
        model_name=settings.text_embedding_model,
        device=settings.text_embedding_device,
    )

    repo = DataRepository(ParquetStore(settings.data_dir))
    playlists_df = repo.load_playlists()
    matches_df, _ = repo.load_labeled_historical_matches()
    tracks_df = repo.load_tracks_from_export(args.tracks_csv)

    print(
        f"[Embeddings] Sources: playlists={len(playlists_df)} "
        f"tracks_with_audio={len(tracks_df)} historical_rows={len(matches_df)}"
    )

    # Hydrate tracks present only in match history so we cache their embeddings too.
    matches_meta = matches_df[
        [c for c in ("track_id", "track_name", "artist") if c in matches_df.columns]
    ].drop_duplicates(subset=["track_id"], keep="last")
    extras = matches_meta[~matches_meta["track_id"].isin(tracks_df["track_id"])]
    if not extras.empty:
        for col in tracks_df.columns:
            if col not in extras.columns:
                extras[col] = None
        import pandas as pd
        tracks_df = pd.concat([tracks_df, extras[tracks_df.columns]], ignore_index=True)
    print(f"[Embeddings] Total unique tracks to embed: {len(tracks_df)}")

    embedder.encode(build_playlist_text_strings(playlists_df))
    embedder.encode(build_track_text_strings(tracks_df))
    print(f"[Embeddings] Cache written to {embedder.cache_path}")


if __name__ == "__main__":
    main()
