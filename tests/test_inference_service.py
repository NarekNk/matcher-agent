from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from matcher_agent.artifacts.io import save_bundle
from matcher_agent.features.feature_builder import PAIRWISE_FEATURE_COLS
from matcher_agent.inference.service import MatcherService
from matcher_agent.models import TrackInput


class _StubEmbedder:
    """Deterministic stub embedder so tests don't load sentence-transformers."""

    model_name = "stub-test"

    def __init__(self) -> None:
        self._lookup: dict[str, np.ndarray] = {}

    def encode(self, texts):
        out = []
        for t in texts:
            t_norm = (t or "").strip().lower()
            if "rap" in t_norm or "hip" in t_norm or "trap" in t_norm:
                vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            elif "country" in t_norm:
                vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            else:
                vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            out.append(vec)
        return np.vstack(out)


def _make_dummy_pipeline(features: list[str]) -> Pipeline:
    """Train on tiny synthetic data so a real, predict_proba-capable pipeline exists."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((40, len(features))), columns=features)
    # Make the label correlate with semantic_similarity so it produces sensible scores.
    y = (X["semantic_similarity"] > X["semantic_similarity"].median()).astype(int)
    pre = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())]), features)]
    )
    pipe = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=200))])
    pipe.fit(X, y)
    return pipe


def test_recommendation_returns_top_n_and_prefers_genre_match(tmp_path: Path) -> None:
    pipe = _make_dummy_pipeline(PAIRWISE_FEATURE_COLS)
    save_bundle({"model": pipe, "feature_columns": PAIRWISE_FEATURE_COLS}, tmp_path)

    historical = pd.DataFrame(
        [
            {"track_id": "t_rap", "playlist_id": "hh", "label": 1, "track_name": "Drill", "artist": "MC"},
            {"track_id": "t_country", "playlist_id": "co", "label": 1, "track_name": "Truck", "artist": "Joe"},
        ]
    )
    playlists = pd.DataFrame(
        [
            {"playlist_id": "hh", "playlist_name": "Hip-Hop Hits", "description": "rap and trap"},
            {"playlist_id": "co", "playlist_name": "Country Roads", "description": "country only"},
        ]
    )
    tracks = pd.DataFrame(
        [
            {"track_id": "t_rap", "track_name": "Drill", "artist": "MC", "bpm": 140},
            {"track_id": "t_country", "track_name": "Truck", "artist": "Joe", "bpm": 90},
        ]
    )

    service = MatcherService(
        artifact_dir=str(tmp_path),
        historical_df=historical,
        playlists_df=playlists,
        tracks_df=tracks,
        text_embedder=_StubEmbedder(),
        hard_genre_filter=True,
    )
    recs = service.recommend_playlists(
        TrackInput(track_id="new_rap", track_name="New Trap Banger", artist="Newcomer"), n=2
    )
    assert len(recs) == 2
    assert recs[0].rank == 1
    # The hip-hop playlist must outrank the country one for a rap track.
    assert recs[0].playlist_id == "hh"
