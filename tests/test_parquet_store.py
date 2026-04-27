from __future__ import annotations

from pathlib import Path

import pandas as pd

from matcher_agent.storage.parquet_store import ParquetStore


def test_parquet_upsert_keeps_latest_by_updated_at(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    first = pd.DataFrame(
        [
            {"match_id": "m1", "updated_at": "2026-01-01T00:00:00", "status": "declined"},
        ]
    )
    second = pd.DataFrame(
        [
            {"match_id": "m1", "updated_at": "2026-01-02T00:00:00", "status": "accepted"},
            {"match_id": "m2", "updated_at": "2026-01-02T00:00:00", "status": "declined"},
        ]
    )
    store.upsert_table("historical_matches", first, key_col="match_id")
    merged = store.upsert_table("historical_matches", second, key_col="match_id")
    assert len(merged) == 2
    row = merged[merged["match_id"] == "m1"].iloc[0]
    assert row["status"] == "accepted"
