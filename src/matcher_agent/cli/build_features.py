from __future__ import annotations

import argparse
from pathlib import Path

from matcher_agent.config import get_settings
from matcher_agent.data.repository import DataRepository
from matcher_agent.pipeline import (
    build_track_feature_export_sync,
    limit_playlist_sources,
    playlist_sources_from_parquet,
)
from matcher_agent.storage.parquet_store import ParquetStore


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build audio feature dataset from Spotify playlists.")
    parser.add_argument("--output-csv", default="output/training_data.csv")
    parser.add_argument("--max-playlists", type=int, default=settings.max_playlists)
    parser.add_argument("--max-tracks-per-playlist", type=int, default=settings.max_tracks_per_playlist)
    parser.add_argument(
        "--download-concurrency",
        type=int,
        default=settings.feature_download_concurrency,
    )
    parser.add_argument(
        "--analysis-workers",
        type=int,
        default=settings.feature_analysis_workers,
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=settings.feature_progress_every,
    )
    args = parser.parse_args()

    if not settings.spotify_client_id or not settings.spotify_client_secret:
        raise ValueError("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in environment.")
    print("[BuildFeatures] Starting feature build.")

    repo = DataRepository(ParquetStore(settings.data_dir))
    playlists_df = repo.load_playlists()
    playlist_sources = playlist_sources_from_parquet(playlists_df)
    if not playlist_sources:
        raise ValueError("No playlists available in local parquet. Run sync_xano first.")
    playlist_sources = limit_playlist_sources(playlist_sources, max_playlists=args.max_playlists)
    print(
        f"[BuildFeatures] Using playlists={len(playlist_sources)} "
        f"max_tracks_per_playlist={args.max_tracks_per_playlist}"
    )

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = Path.cwd() / output_csv

    df = build_track_feature_export_sync(
        playlist_sources=playlist_sources,
        spotify_client_id=settings.spotify_client_id,
        spotify_client_secret=settings.spotify_client_secret,
        preview_resolver_url=settings.preview_resolver_url,
        audio_dir=settings.audio_dir,
        output_csv=output_csv,
        max_tracks_per_playlist=args.max_tracks_per_playlist,
        download_concurrency=args.download_concurrency,
        analysis_workers=args.analysis_workers,
        progress_every=args.progress_every,
    )
    print(f"[BuildFeatures] Completed rows={int(len(df))} output_csv={output_csv}")


if __name__ == "__main__":
    main()
