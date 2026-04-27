from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from matcher_agent.data.label_mapper import map_status_to_label
from matcher_agent.storage.parquet_store import ParquetStore


@dataclass
class DataRepository:
    store: ParquetStore

    def load_playlists(self) -> pd.DataFrame:
        return self.store.read_table("playlists")

    def load_historical_matches(self) -> pd.DataFrame:
        return self.store.read_table("historical_matches")

    def load_labeled_historical_matches(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        historical = self.load_historical_matches()
        if historical.empty:
            return historical, historical
        return map_status_to_label(historical, status_col="status", output_col="label")

    def load_tracks_from_export(self, training_data_csv: str) -> pd.DataFrame:
        df = pd.read_csv(training_data_csv)
        track_cols = [c for c in df.columns if c.startswith("mfcc_")] + [
            "track_id",
            "track_name",
            "artist",
            "album",
            "duration_ms",
            "bpm",
            "beats_confidence",
            "loudness",
            "danceability",
            "energy",
            "spectral_centroid",
            "spectral_rolloff",
            "spectral_flux",
            "zcr",
        ]
        cols = [c for c in track_cols if c in df.columns]
        return df[cols].drop_duplicates(subset=["track_id"]).reset_index(drop=True)
