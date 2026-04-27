from __future__ import annotations

import argparse

from matcher_agent.clients.xano_client import XanoClient
from matcher_agent.config import get_settings
from matcher_agent.storage.parquet_store import ParquetStore
from matcher_agent.sync.xano_sync import XanoSyncConfig, XanoSyncService


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync playlists and historical matches from Xano into local parquet.")
    parser.add_argument("--full-refresh", action="store_true", help="Ignore watermark and overwrite local snapshot.")
    parser.add_argument("--playlist-key-col", default="playlist_id")
    parser.add_argument("--historical-key-col", default="match_id")
    parser.add_argument("--updated-at-col", default="updated_at")
    args = parser.parse_args()

    settings = get_settings()
    print("[Sync] Starting Xano sync.")
    if not settings.xano_playlists_url or not settings.xano_historical_matches_url:
        raise ValueError("Set XANO_PLAYLISTS_URL and XANO_HISTORICAL_MATCHES_URL in environment.")

    client = XanoClient(
        playlists_url=settings.xano_playlists_url,
        historical_matches_url=settings.xano_historical_matches_url,
        timeout_s=settings.xano_timeout_s,
        page_size=settings.xano_page_size,
        historical_max_pages=(
            settings.xano_historical_max_pages if settings.xano_historical_max_pages > 0 else None
        ),
    )
    store = ParquetStore(settings.data_dir)
    sync_service = XanoSyncService(
        client=client,
        store=store,
        config=XanoSyncConfig(
            playlist_key_col=args.playlist_key_col,
            historical_key_col=args.historical_key_col,
            updated_at_col=args.updated_at_col,
        ),
    )
    print(
        "[Sync] Config:",
        {
            "full_refresh": args.full_refresh,
            "page_size": settings.xano_page_size,
            "historical_max_pages": settings.xano_historical_max_pages,
        },
    )
    result = sync_service.sync(full_refresh=args.full_refresh)
    print(f"[Sync] Completed: {result}")


if __name__ == "__main__":
    main()
