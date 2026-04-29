from __future__ import annotations

import numpy as np
import pandas as pd

from matcher_agent.features.feature_builder import (
    PAIRWISE_FEATURE_COLS,
    build_pair_features,
    build_track_audio_lookup,
    build_track_meta_lookup,
    select_model_features,
)
from matcher_agent.features.playlist_profiles import build_profiles


def _emb(*v: float) -> np.ndarray:
    return np.asarray(v, dtype=np.float32)


def _build_minimal_bundle():
    playlists = pd.DataFrame(
        [
            {"playlist_id": "hh", "playlist_name": "Hip-Hop Hits", "description": "rap and trap"},
            {"playlist_id": "co", "playlist_name": "Country Roads", "description": "country only"},
        ]
    )
    matches = pd.DataFrame(
        [
            {"playlist_id": "hh", "track_id": "t_rap", "label": 1},
            {"playlist_id": "co", "track_id": "t_country", "label": 1},
        ]
    )
    tracks = pd.DataFrame(
        [
            {
                "track_id": "t_rap",
                "artist": "MC Foo",
                "track_name": "Drill Anthem",
                "bpm": 140,
                "loudness": -6,
                "popularity": 70,
            },
            {
                "track_id": "t_country",
                "artist": "Country Joe",
                "track_name": "Pickup Truck",
                "bpm": 90,
                "loudness": -10,
                "popularity": 25,
            },
        ]
    )
    track_emb = {
        "t_rap": _emb(1.0, 0.0, 0.0),
        "t_country": _emb(0.0, 1.0, 0.0),
    }
    playlist_emb = {
        "hh": _emb(1.0, 0.0, 0.0),
        "co": _emb(0.0, 1.0, 0.0),
    }
    bundle = build_profiles(
        playlists,
        matches,
        tracks,
        track_text_emb_by_id=track_emb,
        playlist_text_emb_by_id=playlist_emb,
    )
    return bundle, tracks, track_emb


def test_pair_features_capture_genre_alignment() -> None:
    bundle, tracks, track_emb = _build_minimal_bundle()
    audio_lookup = build_track_audio_lookup(tracks, bundle.audio_feature_cols)
    meta_lookup = build_track_meta_lookup(tracks)

    pairs = pd.DataFrame(
        [
            {"track_id": "t_rap", "playlist_id": "hh", "label": 1},
            {"track_id": "t_rap", "playlist_id": "co", "label": 0},
            {"track_id": "t_country", "playlist_id": "co", "label": 1},
            {"track_id": "t_country", "playlist_id": "hh", "label": 0},
        ]
    )
    feats = build_pair_features(
        pairs,
        profile_bundle=bundle,
        track_text_emb_by_id=track_emb,
        track_audio_by_id=audio_lookup,
        track_meta_by_id=meta_lookup,
    )

    rap_to_hh = feats[(feats.track_id == "t_rap") & (feats.playlist_id == "hh")].iloc[0]
    rap_to_co = feats[(feats.track_id == "t_rap") & (feats.playlist_id == "co")].iloc[0]

    # Same-genre pair must score higher on semantic similarity than cross-genre.
    assert rap_to_hh["semantic_similarity"] > rap_to_co["semantic_similarity"]
    # Genre conflict flag should fire on rap → country.
    assert rap_to_co["genre_conflict_flag"] == 1.0
    assert rap_to_hh["genre_conflict_flag"] == 0.0
    # Genre overlap count should be > 0 on the matching pair.
    assert rap_to_hh["genre_overlap_count"] >= 1.0


def test_pair_features_compute_popularity_fit() -> None:
    bundle, tracks, track_emb = _build_minimal_bundle()
    audio_lookup = build_track_audio_lookup(tracks, bundle.audio_feature_cols)
    meta_lookup = build_track_meta_lookup(tracks)
    pop_lookup = {"t_rap": 70.0, "t_country": 25.0}

    pairs = pd.DataFrame(
        [
            {"track_id": "t_rap", "playlist_id": "hh", "label": 1},
            {"track_id": "t_country", "playlist_id": "co", "label": 1},
        ]
    )
    feats = build_pair_features(
        pairs,
        profile_bundle=bundle,
        track_text_emb_by_id=track_emb,
        track_audio_by_id=audio_lookup,
        track_meta_by_id=meta_lookup,
        track_popularity_by_id=pop_lookup,
    )

    # Each playlist has exactly one accepted track, so playlist_popularity_mean
    # equals the track's popularity → diff_norm should be ~0 and
    # popularity_available should be 1 for both pairs.
    rap = feats[(feats.track_id == "t_rap") & (feats.playlist_id == "hh")].iloc[0]
    cnt = feats[(feats.track_id == "t_country") & (feats.playlist_id == "co")].iloc[0]
    assert rap["popularity_available"] == 1.0
    assert cnt["popularity_available"] == 1.0
    assert abs(rap["popularity_diff_norm"]) < 1e-9
    assert abs(cnt["popularity_diff_norm"]) < 1e-9
    # Track popularity is exposed normalized to [0, 1].
    assert 0.69 < rap["track_popularity_norm"] < 0.71
    assert 0.24 < cnt["track_popularity_norm"] < 0.26


def test_pair_features_handle_missing_popularity() -> None:
    bundle, tracks, track_emb = _build_minimal_bundle()
    audio_lookup = build_track_audio_lookup(tracks, bundle.audio_feature_cols)
    meta_lookup = build_track_meta_lookup(tracks)

    pairs = pd.DataFrame([{"track_id": "t_rap", "playlist_id": "hh", "label": 1}])
    feats = build_pair_features(
        pairs,
        profile_bundle=bundle,
        track_text_emb_by_id=track_emb,
        track_audio_by_id=audio_lookup,
        track_meta_by_id=meta_lookup,
        track_popularity_by_id={},  # no popularity at all
    )
    row = feats.iloc[0]
    assert row["popularity_available"] == 0.0
    assert row["track_popularity_norm"] == 0.0
    assert row["popularity_diff_norm"] == 0.0
    assert row["popularity_zscore"] == 0.0


def test_select_model_features_returns_pairwise_columns_only() -> None:
    df = pd.DataFrame(columns=[*PAIRWISE_FEATURE_COLS, "track_id", "playlist_id", "label"])
    cols = select_model_features(df)
    assert set(cols) == set(PAIRWISE_FEATURE_COLS)
    assert "track_id" not in cols
    assert "label" not in cols
    # Ensure the new popularity features are part of the model contract.
    for must_have in (
        "track_popularity_norm",
        "popularity_diff_norm",
        "popularity_zscore",
        "popularity_available",
    ):
        assert must_have in cols
