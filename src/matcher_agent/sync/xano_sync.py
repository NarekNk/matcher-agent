from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from matcher_agent.clients.xano_client import XanoClient
from matcher_agent.storage.parquet_store import ParquetStore


@dataclass
class XanoSyncConfig:
    playlist_table_name: str = "playlists"
    historical_table_name: str = "historical_matches"
    playlist_key_col: str = "playlist_id"
    historical_key_col: str = "match_id"
    updated_at_col: str = "updated_at"


class XanoSyncService:
    def __init__(self, client: XanoClient, store: ParquetStore, config: XanoSyncConfig | None = None):
        self.client = client
        self.store = store
        self.config = config or XanoSyncConfig()

    def _table_watermark(self, state: dict[str, Any], table_name: str) -> str | None:
        return state.get(table_name, {}).get("watermark")

    def _update_state_watermark(self, state: dict[str, Any], table_name: str, df: pd.DataFrame) -> None:
        if self.config.updated_at_col not in df.columns or df.empty:
            return
        watermark = pd.to_datetime(df[self.config.updated_at_col], errors="coerce").max()
        if pd.notna(watermark):
            state.setdefault(table_name, {})["watermark"] = watermark.isoformat()

    def sync(self, *, full_refresh: bool = False) -> dict[str, int]:
        print(f"[SyncService] sync started (full_refresh={full_refresh})")
        state = self.store.read_state()
        playlists_after = None if full_refresh else self._table_watermark(state, self.config.playlist_table_name)
        history_after = None if full_refresh else self._table_watermark(state, self.config.historical_table_name)
        print(
            "[SyncService] watermarks:",
            {"playlists_after": playlists_after, "history_after": history_after},
        )

        playlists_rows = self.client.fetch_playlists(updated_after=playlists_after)
        historical_rows = self.client.fetch_historical_matches(updated_after=history_after)
        print(
            "[SyncService] fetched rows:",
            {"playlists_rows": len(playlists_rows), "historical_rows": len(historical_rows)},
        )
        playlists_df = pd.DataFrame(playlists_rows)
        historical_df = pd.DataFrame(historical_rows)

        if full_refresh:
            if not playlists_df.empty:
                self.store.write_table(self.config.playlist_table_name, playlists_df)
            if not historical_df.empty:
                self.store.write_table(self.config.historical_table_name, historical_df)
        else:
            if not playlists_df.empty:
                playlists_df = self.store.upsert_table(
                    self.config.playlist_table_name,
                    playlists_df,
                    key_col=self.config.playlist_key_col,
                    updated_at_col=self.config.updated_at_col,
                )
            else:
                playlists_df = self.store.read_table(self.config.playlist_table_name)

            if not historical_df.empty:
                historical_df = self.store.upsert_table(
                    self.config.historical_table_name,
                    historical_df,
                    key_col=self.config.historical_key_col,
                    updated_at_col=self.config.updated_at_col,
                )
            else:
                historical_df = self.store.read_table(self.config.historical_table_name)

        self._update_state_watermark(state, self.config.playlist_table_name, playlists_df)
        self._update_state_watermark(state, self.config.historical_table_name, historical_df)
        self.store.write_state(state)
        print("[SyncService] sync state persisted.")

        return {
            "playlists_rows": int(len(playlists_df)),
            "historical_rows": int(len(historical_df)),
        }
