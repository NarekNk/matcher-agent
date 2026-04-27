from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


class ParquetStore:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root_dir / "sync_state.json"

    def table_path(self, name: str) -> Path:
        return self.root_dir / f"{name}.parquet"

    def read_table(self, name: str) -> pd.DataFrame:
        path = self.table_path(name)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def write_table(self, name: str, df: pd.DataFrame) -> Path:
        path = self.table_path(name)
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)
        return path

    def upsert_table(
        self,
        name: str,
        incoming_df: pd.DataFrame,
        *,
        key_col: str,
        updated_at_col: str = "updated_at",
    ) -> pd.DataFrame:
        if incoming_df.empty:
            return self.read_table(name)

        existing = self.read_table(name)
        merged = pd.concat([existing, incoming_df], ignore_index=True)
        if updated_at_col in merged.columns:
            merged[updated_at_col] = pd.to_datetime(merged[updated_at_col], errors="coerce")
            merged = merged.sort_values(updated_at_col)
        merged = merged.drop_duplicates(subset=[key_col], keep="last").reset_index(drop=True)
        self.write_table(name, merged)
        return merged

    def read_state(self) -> dict:
        if not self.state_path.exists():
            return {}
        return json.loads(self.state_path.read_text())

    def write_state(self, state: dict) -> None:
        self.state_path.write_text(json.dumps(state, indent=2, sort_keys=True))
