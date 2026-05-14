"""Tests for audio analysis improvements: temporal features, key/mode,
onset rate, AUDIO_FEATURE_COLS expansion, and backward compatibility."""

from __future__ import annotations

import numpy as np
import pandas as pd

from matcher_agent.audio.analyzer import _key_to_int
from matcher_agent.features.playlist_profiles import (
    AUDIO_FEATURE_COLS,
    ensure_audio_columns,
    build_profiles,
)
from matcher_agent.features.feature_builder import (
    build_pair_features,
    build_track_audio_lookup,
    build_track_meta_lookup,
)


def _emb(*v: float) -> np.ndarray:
    return np.asarray(v, dtype=np.float32)


# ---- _key_to_int tests ----


def test_key_to_int_maps_all_keys() -> None:
    expected = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
        "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
        "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
    }
    for key_str, expected_int in expected.items():
        assert _key_to_int(key_str) == expected_int


def test_key_to_int_unknown_returns_minus_one() -> None:
    assert _key_to_int("X") == -1
    assert _key_to_int("") == -1


# ---- AUDIO_FEATURE_COLS tests ----


def test_audio_feature_cols_contains_original_columns() -> None:
    original = [
        "bpm", "loudness", "danceability", "energy",
        "spectral_centroid", "spectral_rolloff", "spectral_flux", "zcr",
        *(f"mfcc_{i}" for i in range(1, 14)),
    ]
    for col in original:
        assert col in AUDIO_FEATURE_COLS, f"Missing original column: {col}"


def test_audio_feature_cols_contains_new_columns() -> None:
    new_cols = [
        "key", "mode", "onset_rate",
        "spectral_centroid_std", "spectral_rolloff_std",
        "spectral_flux_std", "zcr_std",
        *(f"mfcc_{i}_std" for i in range(1, 14)),
    ]
    for col in new_cols:
        assert col in AUDIO_FEATURE_COLS, f"Missing new column: {col}"


def test_audio_feature_cols_new_columns_at_end() -> None:
    """New columns must come after the original 21 to preserve alignment
    when loading old cached data."""
    original_count = 21
    original_block = AUDIO_FEATURE_COLS[:original_count]
    assert "bpm" == original_block[0]
    assert "mfcc_13" == original_block[-1]
    assert "key" in AUDIO_FEATURE_COLS[original_count:]


def test_audio_feature_cols_total_count() -> None:
    assert len(AUDIO_FEATURE_COLS) == 41


# ---- ensure_audio_columns backward compat tests ----


def test_ensure_audio_columns_fills_missing() -> None:
    """Old CSV with only the original 21 cols gets the new ones filled with NaN."""
    original_cols = list(AUDIO_FEATURE_COLS[:21])
    df = pd.DataFrame(
        [{col: float(i) for i, col in enumerate(original_cols)}],
    )
    assert "key" not in df.columns
    result = ensure_audio_columns(df)

    for col in AUDIO_FEATURE_COLS:
        assert col in result.columns, f"Missing column after migration: {col}"

    row = result.iloc[0]
    assert row["bpm"] == 0.0
    assert np.isnan(row["key"])
    assert np.isnan(row["onset_rate"])
    assert np.isnan(row["spectral_centroid_std"])
    assert np.isnan(row["mfcc_1_std"])


def test_ensure_audio_columns_noop_when_all_present() -> None:
    """If all columns already exist, the function is a no-op."""
    df = pd.DataFrame(
        [{col: float(i) for i, col in enumerate(AUDIO_FEATURE_COLS)}],
    )
    result = ensure_audio_columns(df)
    assert list(result.columns) == list(df.columns)
    assert result.iloc[0]["key"] == float(list(AUDIO_FEATURE_COLS).index("key"))


def test_ensure_audio_columns_preserves_extra_columns() -> None:
    """Non-audio columns (track_id etc.) survive the migration."""
    df = pd.DataFrame(
        [{"track_id": "t1", "bpm": 120.0, "loudness": -8.0}],
    )
    result = ensure_audio_columns(df)
    assert "track_id" in result.columns
    assert result.iloc[0]["track_id"] == "t1"
    assert np.isnan(result.iloc[0]["key"])


# ---- build_track_audio_lookup with new cols ----


def _make_tracks_df_v2():
    """Tracks DataFrame with all v2 audio columns populated."""
    row = {"track_id": "t1", "artist": "A", "track_name": "Test"}
    for i, col in enumerate(AUDIO_FEATURE_COLS):
        row[col] = float(i)
    return pd.DataFrame([row])


def test_audio_lookup_includes_new_dimensions() -> None:
    tracks_df = _make_tracks_df_v2()
    lookup = build_track_audio_lookup(tracks_df, list(AUDIO_FEATURE_COLS))
    assert "t1" in lookup
    vec = lookup["t1"]
    assert vec.shape == (len(AUDIO_FEATURE_COLS),)
    assert vec[0] == 0.0  # bpm
    key_idx = list(AUDIO_FEATURE_COLS).index("key")
    assert vec[key_idx] == float(key_idx)


def test_audio_lookup_handles_nan_new_columns() -> None:
    """Old tracks missing new columns (filled with NaN) should still be loaded
    if their original columns are valid."""
    original = list(AUDIO_FEATURE_COLS[:21])
    row = {"track_id": "t1"}
    for col in original:
        row[col] = 1.0
    df = pd.DataFrame([row])
    df = ensure_audio_columns(df)
    lookup = build_track_audio_lookup(df, list(AUDIO_FEATURE_COLS))
    assert "t1" in lookup
    vec = lookup["t1"]
    assert not np.isnan(vec[0])
    key_idx = list(AUDIO_FEATURE_COLS).index("key")
    assert np.isnan(vec[key_idx])


# ---- Profile building with new audio columns ----


def test_profiles_audio_centroid_includes_new_dims() -> None:
    """Profile audio centroids should have a length equal to the number of
    available audio columns, including the new v2 ones."""
    tracks_df = _make_tracks_df_v2()
    playlists = pd.DataFrame(
        [{"playlist_id": "p1", "playlist_name": "Test", "description": ""}]
    )
    matches = pd.DataFrame(
        [{"playlist_id": "p1", "track_id": "t1", "label": 1}]
    )
    bundle = build_profiles(
        playlists, matches, tracks_df,
        track_text_emb_by_id={"t1": _emb(1.0, 0.0)},
        playlist_text_emb_by_id={"p1": _emb(0.0, 1.0)},
    )
    prof = bundle.profiles["p1"]
    assert prof.audio_centroid is not None
    assert len(prof.audio_centroid) == len(AUDIO_FEATURE_COLS)


# ---- Feature builder with new audio dims ----


def test_pair_features_work_with_expanded_audio() -> None:
    """build_pair_features should not error with the expanded audio vectors."""
    tracks_df = _make_tracks_df_v2()
    # Add a second track for cross-pairing
    row2 = {"track_id": "t2", "artist": "B", "track_name": "Other"}
    for i, col in enumerate(AUDIO_FEATURE_COLS):
        row2[col] = float(i) + 10.0
    tracks_df = pd.concat([tracks_df, pd.DataFrame([row2])], ignore_index=True)

    playlists = pd.DataFrame(
        [{"playlist_id": "p1", "playlist_name": "Test", "description": ""}]
    )
    matches = pd.DataFrame(
        [
            {"playlist_id": "p1", "track_id": "t1", "label": 1},
            {"playlist_id": "p1", "track_id": "t2", "label": 1},
        ]
    )
    track_emb = {
        "t1": _emb(1.0, 0.0),
        "t2": _emb(0.0, 1.0),
    }
    playlist_emb = {"p1": _emb(0.5, 0.5)}
    bundle = build_profiles(
        playlists, matches, tracks_df,
        track_text_emb_by_id=track_emb,
        playlist_text_emb_by_id=playlist_emb,
    )
    audio_lookup = build_track_audio_lookup(tracks_df, bundle.audio_feature_cols)
    meta_lookup = build_track_meta_lookup(tracks_df)

    pairs = pd.DataFrame(
        [{"track_id": "t1", "playlist_id": "p1", "label": 1}]
    )
    feats = build_pair_features(
        pairs,
        profile_bundle=bundle,
        track_text_emb_by_id=track_emb,
        track_audio_by_id=audio_lookup,
        track_meta_by_id=meta_lookup,
    )
    assert len(feats) == 1
    row = feats.iloc[0]
    assert row["audio_centroid_cosine"] != 0.0
    assert row["audio_zscore_mean"] >= 0.0
    assert row["audio_range_ratio"] >= 0.0


def test_pair_features_with_mixed_v1_v2_tracks() -> None:
    """A v1 track (NaN on new cols) paired against a v2 profile should
    compute features without error; the NaN dimensions are handled."""
    v2_row = {"track_id": "t_full", "artist": "A", "track_name": "Full"}
    for i, col in enumerate(AUDIO_FEATURE_COLS):
        v2_row[col] = float(i)
    v1_row = {"track_id": "t_old"}
    for col in AUDIO_FEATURE_COLS[:21]:
        v1_row[col] = 5.0

    df = pd.DataFrame([v2_row, v1_row])
    df = ensure_audio_columns(df)

    playlists = pd.DataFrame(
        [{"playlist_id": "p1", "playlist_name": "Test", "description": ""}]
    )
    matches = pd.DataFrame(
        [{"playlist_id": "p1", "track_id": "t_full", "label": 1}]
    )
    track_emb = {
        "t_full": _emb(1.0, 0.0),
        "t_old": _emb(0.0, 1.0),
    }
    playlist_emb = {"p1": _emb(0.5, 0.5)}
    bundle = build_profiles(
        playlists, matches, df,
        track_text_emb_by_id=track_emb,
        playlist_text_emb_by_id=playlist_emb,
    )
    audio_lookup = build_track_audio_lookup(df, bundle.audio_feature_cols)
    meta_lookup = build_track_meta_lookup(df)

    pairs = pd.DataFrame(
        [{"track_id": "t_old", "playlist_id": "p1", "label": 0}]
    )
    feats = build_pair_features(
        pairs,
        profile_bundle=bundle,
        track_text_emb_by_id=track_emb,
        track_audio_by_id=audio_lookup,
        track_meta_by_id=meta_lookup,
    )
    assert len(feats) == 1
    row = feats.iloc[0]
    assert row["track_audio_available"] == 1.0
    assert row["audio_centroid_cosine"] != 0.0
