from __future__ import annotations

import argparse
import asyncio
import json

from matcher_agent.audio.analyzer import analyze_audio
from matcher_agent.audio.downloader import download_preview
from matcher_agent.clients.preview_resolver_client import enrich_tracks_preview_urls
from matcher_agent.clients.spotify_client import fetch_track_by_id, get_spotify_client
from matcher_agent.config import get_settings
from matcher_agent.data.repository import DataRepository
from matcher_agent.embeddings import TextEmbedder
from matcher_agent.inference.service import MatcherService
from matcher_agent.models import TrackInput
from matcher_agent.storage.parquet_store import ParquetStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend top-N playlists for a track.")
    parser.add_argument("--spotify-track-id", required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--tracks-csv", default="output/training_data.csv")
    parser.add_argument(
        "--no-genre-filter",
        action="store_true",
        help="Disable hard genre conflict filtering (returns raw GBM scores).",
    )
    args = parser.parse_args()

    print("[RecommendCLI] Starting recommendation request.")
    settings = get_settings()
    if not settings.spotify_client_id or not settings.spotify_client_secret:
        raise ValueError("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in environment.")

    print(f"[RecommendCLI] Fetching Spotify metadata for track_id={args.spotify_track_id}")
    sp = get_spotify_client(settings.spotify_client_id, settings.spotify_client_secret)
    track_meta = fetch_track_by_id(sp, args.spotify_track_id)
    resolved = asyncio.run(enrich_tracks_preview_urls([track_meta], settings.preview_resolver_url))[0]
    print(
        f"[RecommendCLI] Track fetched name='{resolved.get('track_name')}' "
        f"artist='{resolved.get('artist')}' preview={'yes' if resolved.get('preview_url') else 'no'}"
    )

    extra_features: dict = {}
    preview_url = resolved.get("preview_url")
    if preview_url:
        audio_path = download_preview(resolved["track_id"], preview_url, settings.audio_dir)
        if audio_path:
            audio_features = analyze_audio(audio_path)
            if audio_features:
                extra_features.update(audio_features)
                extra_features["audio_path"] = str(audio_path)
                print(f"[RecommendCLI] Audio analysis completed features={len(audio_features)}")
            else:
                print("[RecommendCLI] Audio analysis failed; scoring without audio features.")
        else:
            print("[RecommendCLI] Preview download failed; scoring without audio features.")
    else:
        print("[RecommendCLI] No preview URL available; scoring without audio features.")

    repo = DataRepository(ParquetStore(settings.data_dir))
    historical_df, _ = repo.load_labeled_historical_matches()
    playlists_df = repo.load_playlists()
    tracks_df = repo.load_tracks_from_export(args.tracks_csv)
    print(
        f"[RecommendCLI] Loaded playlists={len(playlists_df)} "
        f"historical_rows={len(historical_df)} tracks={len(tracks_df)}"
    )

    embedder = TextEmbedder(
        cache_path=settings.embeddings_dir / "text_embeddings.parquet",
        model_name=settings.text_embedding_model,
        device=settings.text_embedding_device,
    )
    service = MatcherService(
        artifact_dir=str(settings.model_dir),
        historical_df=historical_df,
        playlists_df=playlists_df,
        tracks_df=tracks_df,
        text_embedder=embedder,
        semantic_blend=settings.semantic_blend,
        hard_genre_filter=not args.no_genre_filter and settings.hard_genre_filter,
    )
    recs = service.recommend_playlists(
        TrackInput(
            track_id=resolved.get("track_id"),
            track_name=resolved.get("track_name", ""),
            artist=resolved.get("artist", ""),
            album=resolved.get("album"),
            duration_ms=resolved.get("duration_ms"),
            preview_url=resolved.get("preview_url"),
            spotify_url=resolved.get("spotify_url"),
            artist_genres=resolved.get("artist_genres") or [],
            popularity=resolved.get("popularity"),
            extra=extra_features,
        ),
        n=args.n,
    )
    print("[RecommendCLI] Completed.")
    print(json.dumps([r.__dict__ for r in recs], indent=2))


if __name__ == "__main__":
    main()
