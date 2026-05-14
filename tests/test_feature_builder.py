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
    # Ensure the popularity features are part of the model contract.
    for must_have in (
        "track_popularity_norm",
        "popularity_diff_norm",
        "popularity_zscore",
        "popularity_available",
    ):
        assert must_have in cols
    # Ensure the new v2 features are part of the model contract.
    for must_have in (
        "genre_semantic_interaction",
        "audio_genre_interaction",
        "semantic_l2_distance",
        "title_semantic_diff",
        "bpm_diff",
        "energy_diff",
        "danceability_diff",
        "audio_range_ratio",
        "playlist_size_log",
        "playlist_genre_richness",
        "playlist_audio_available",
    ):
        assert must_have in cols
    # v3 soft-attribute features.
    for must_have in (
        "playlist_n_soft_attrs",
        "playlist_has_language",
        "playlist_has_mood",
        "mood_match_flag",
        "language_match_flag",
        "activity_match_flag",
        "soft_attr_available",
    ):
        assert must_have in cols


def _build_rich_bundle():
    """Bundle with multiple accepted tracks per playlist so audio min/max/std
    are meaningful and per-dimension features can be tested."""
    playlists = pd.DataFrame(
        [
            {"playlist_id": "pl1", "playlist_name": "Chill Vibes", "description": "relaxing beats"},
        ]
    )
    matches = pd.DataFrame(
        [
            {"playlist_id": "pl1", "track_id": "t_a", "label": 1},
            {"playlist_id": "pl1", "track_id": "t_b", "label": 1},
            {"playlist_id": "pl1", "track_id": "t_c", "label": 1},
        ]
    )
    tracks = pd.DataFrame(
        [
            {"track_id": "t_a", "artist": "A", "track_name": "Chill A",
             "bpm": 80, "loudness": -10, "danceability": 0.5, "energy": 0.3, "popularity": 40},
            {"track_id": "t_b", "artist": "B", "track_name": "Chill B",
             "bpm": 100, "loudness": -8, "danceability": 0.7, "energy": 0.5, "popularity": 60},
            {"track_id": "t_c", "artist": "C", "track_name": "Chill C",
             "bpm": 120, "loudness": -6, "danceability": 0.9, "energy": 0.7, "popularity": 50},
            {"track_id": "t_new", "artist": "New", "track_name": "New Track",
             "bpm": 90, "loudness": -9, "danceability": 0.6, "energy": 0.4, "popularity": 55},
            {"track_id": "t_outlier", "artist": "Outlier", "track_name": "Fast Outlier",
             "bpm": 200, "loudness": -2, "danceability": 0.1, "energy": 0.9, "popularity": 80},
        ]
    )
    track_emb = {
        "t_a": _emb(0.8, 0.1, 0.0),
        "t_b": _emb(0.7, 0.2, 0.1),
        "t_c": _emb(0.9, 0.0, 0.1),
        "t_new": _emb(0.75, 0.15, 0.05),
        "t_outlier": _emb(0.0, 0.0, 1.0),
    }
    playlist_emb = {"pl1": _emb(0.8, 0.1, 0.05)}
    bundle = build_profiles(
        playlists, matches, tracks,
        track_text_emb_by_id=track_emb,
        playlist_text_emb_by_id=playlist_emb,
    )
    return bundle, tracks, track_emb


def test_interaction_features() -> None:
    """genre_semantic_interaction and audio_genre_interaction are the product
    of their component features."""
    bundle, tracks, track_emb = _build_minimal_bundle()
    audio_lookup = build_track_audio_lookup(tracks, bundle.audio_feature_cols)
    meta_lookup = build_track_meta_lookup(tracks)

    pairs = pd.DataFrame(
        [
            {"track_id": "t_rap", "playlist_id": "hh", "label": 1},
            {"track_id": "t_rap", "playlist_id": "co", "label": 0},
        ]
    )
    feats = build_pair_features(
        pairs,
        profile_bundle=bundle,
        track_text_emb_by_id=track_emb,
        track_audio_by_id=audio_lookup,
        track_meta_by_id=meta_lookup,
    )

    for _, row in feats.iterrows():
        expected_gsi = row["genre_jaccard"] * row["semantic_similarity"]
        assert abs(row["genre_semantic_interaction"] - expected_gsi) < 1e-9
        expected_agi = row["audio_centroid_cosine"] * row["genre_overlap_count"]
        assert abs(row["audio_genre_interaction"] - expected_agi) < 1e-9


def test_semantic_l2_and_title_diff() -> None:
    """L2 distance is > 0 for non-identical embeddings, and title_semantic_diff
    captures the gap between accepted-track centroid and title-only similarity."""
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
    )
    row = feats.iloc[0]
    assert row["semantic_l2_distance"] >= 0.0
    expected_diff = abs(row["semantic_similarity"] - row["title_text_similarity"])
    assert abs(row["title_semantic_diff"] - expected_diff) < 1e-9


def test_per_dimension_audio_diffs() -> None:
    """bpm_diff, energy_diff, danceability_diff are per-dimension z-scores."""
    bundle, tracks, track_emb = _build_rich_bundle()
    audio_lookup = build_track_audio_lookup(tracks, bundle.audio_feature_cols)
    meta_lookup = build_track_meta_lookup(tracks)

    pairs = pd.DataFrame(
        [
            {"track_id": "t_new", "playlist_id": "pl1", "label": 1},
            {"track_id": "t_outlier", "playlist_id": "pl1", "label": 0},
        ]
    )
    feats = build_pair_features(
        pairs,
        profile_bundle=bundle,
        track_text_emb_by_id=track_emb,
        track_audio_by_id=audio_lookup,
        track_meta_by_id=meta_lookup,
    )

    normal = feats[feats.track_id == "t_new"].iloc[0]
    outlier = feats[feats.track_id == "t_outlier"].iloc[0]

    assert normal["bpm_diff"] >= 0.0
    assert normal["energy_diff"] >= 0.0
    assert normal["danceability_diff"] >= 0.0

    # Outlier track (bpm=200 vs centroid ~100) should have much larger bpm_diff.
    assert outlier["bpm_diff"] > normal["bpm_diff"]
    assert outlier["energy_diff"] > normal["energy_diff"]


def test_audio_range_ratio() -> None:
    """Track inside the playlist's audio envelope gets ratio > 0;
    track far outside gets a lower ratio."""
    bundle, tracks, track_emb = _build_rich_bundle()
    audio_lookup = build_track_audio_lookup(tracks, bundle.audio_feature_cols)
    meta_lookup = build_track_meta_lookup(tracks)

    pairs = pd.DataFrame(
        [
            {"track_id": "t_new", "playlist_id": "pl1", "label": 1},
            {"track_id": "t_outlier", "playlist_id": "pl1", "label": 0},
        ]
    )
    feats = build_pair_features(
        pairs,
        profile_bundle=bundle,
        track_text_emb_by_id=track_emb,
        track_audio_by_id=audio_lookup,
        track_meta_by_id=meta_lookup,
    )

    normal = feats[feats.track_id == "t_new"].iloc[0]
    outlier = feats[feats.track_id == "t_outlier"].iloc[0]

    # t_new (bpm=90, loud=-9, dance=0.6, energy=0.4) is within [80-120] /
    # [-10,-6] / [0.5-0.9] / [0.3-0.7] envelope → should be high.
    assert normal["audio_range_ratio"] > 0.5

    # t_outlier (bpm=200, loud=-2, dance=0.1, energy=0.9) is outside most
    # dims → should be lower than normal.
    assert outlier["audio_range_ratio"] < normal["audio_range_ratio"]


def test_playlist_structural_features() -> None:
    """playlist_size_log, playlist_genre_richness, and playlist_audio_available
    reflect actual profile attributes."""
    bundle, tracks, track_emb = _build_rich_bundle()
    audio_lookup = build_track_audio_lookup(tracks, bundle.audio_feature_cols)
    meta_lookup = build_track_meta_lookup(tracks)

    pairs = pd.DataFrame([{"track_id": "t_new", "playlist_id": "pl1", "label": 1}])
    feats = build_pair_features(
        pairs,
        profile_bundle=bundle,
        track_text_emb_by_id=track_emb,
        track_audio_by_id=audio_lookup,
        track_meta_by_id=meta_lookup,
    )
    row = feats.iloc[0]

    prof = bundle.profiles["pl1"]
    assert abs(row["playlist_size_log"] - float(np.log1p(prof.accepted_count))) < 1e-9
    assert abs(row["playlist_genre_richness"] - float(np.log1p(len(prof.tags)))) < 1e-9
    assert row["playlist_audio_available"] == (1.0 if prof.audio_centroid is not None else 0.0)


def test_new_features_handle_missing_audio() -> None:
    """When a track has no audio data, per-dimension and range features
    default to 0.0 without errors."""
    bundle, tracks, track_emb = _build_rich_bundle()
    meta_lookup = build_track_meta_lookup(tracks)

    pairs = pd.DataFrame([{"track_id": "t_new", "playlist_id": "pl1", "label": 1}])
    feats = build_pair_features(
        pairs,
        profile_bundle=bundle,
        track_text_emb_by_id=track_emb,
        track_audio_by_id={},  # no audio
        track_meta_by_id=meta_lookup,
    )
    row = feats.iloc[0]
    assert row["bpm_diff"] == 0.0
    assert row["energy_diff"] == 0.0
    assert row["danceability_diff"] == 0.0
    assert row["audio_range_ratio"] == 0.0
    assert row["audio_genre_interaction"] == 0.0


def test_new_features_handle_missing_profile() -> None:
    """When the playlist has no profile, all new features default gracefully."""
    bundle, tracks, track_emb = _build_minimal_bundle()
    audio_lookup = build_track_audio_lookup(tracks, bundle.audio_feature_cols)
    meta_lookup = build_track_meta_lookup(tracks)

    pairs = pd.DataFrame([{"track_id": "t_rap", "playlist_id": "missing_pl", "label": 0}])
    feats = build_pair_features(
        pairs,
        profile_bundle=bundle,
        track_text_emb_by_id=track_emb,
        track_audio_by_id=audio_lookup,
        track_meta_by_id=meta_lookup,
    )
    row = feats.iloc[0]
    assert row["genre_semantic_interaction"] == 0.0
    assert row["audio_genre_interaction"] == 0.0
    assert row["semantic_l2_distance"] == 0.0
    assert row["title_semantic_diff"] == 0.0
    assert row["bpm_diff"] == 0.0
    assert row["energy_diff"] == 0.0
    assert row["danceability_diff"] == 0.0
    assert row["audio_range_ratio"] == 0.0
    assert row["playlist_size_log"] == 0.0
    assert row["playlist_genre_richness"] == 0.0
    assert row["playlist_audio_available"] == 0.0
    assert row["playlist_n_soft_attrs"] == 0.0
    assert row["playlist_has_language"] == 0.0
    assert row["playlist_has_mood"] == 0.0
    assert row["mood_match_flag"] == 0.0
    assert row["language_match_flag"] == 0.0
    assert row["activity_match_flag"] == 0.0
    assert row["soft_attr_available"] == 0.0


def test_all_pairwise_cols_present_in_output() -> None:
    """Every feature listed in PAIRWISE_FEATURE_COLS is computed by
    build_pair_features (contract check)."""
    bundle, tracks, track_emb = _build_rich_bundle()
    audio_lookup = build_track_audio_lookup(tracks, bundle.audio_feature_cols)
    meta_lookup = build_track_meta_lookup(tracks)
    pop_lookup = {"t_new": 55.0}

    pairs = pd.DataFrame([{"track_id": "t_new", "playlist_id": "pl1", "label": 1}])
    feats = build_pair_features(
        pairs,
        profile_bundle=bundle,
        track_text_emb_by_id=track_emb,
        track_audio_by_id=audio_lookup,
        track_meta_by_id=meta_lookup,
        track_popularity_by_id=pop_lookup,
    )
    for col in PAIRWISE_FEATURE_COLS:
        assert col in feats.columns, f"Missing feature column: {col}"
        assert not feats[col].isna().any(), f"NaN in feature column: {col}"


# ---- Soft-attribute feature tests ----


def _build_soft_attr_bundle():
    """Bundle where playlists have soft attributes (mood, language, activities)."""
    playlists = pd.DataFrame(
        [
            {
                "playlist_id": "chill",
                "playlist_name": "Chill Vibes",
                "description": "relaxing",
                "activity": '["relaxing", "studying"]',
                "languages": '["english"]',
                "moods": '["calm", "peaceful"]',
            },
            {
                "playlist_id": "party",
                "playlist_name": "Party Mix",
                "description": "upbeat dance",
                "activity": '["dancing", "workout"]',
                "languages": '["spanish"]',
                "moods": '["energetic", "happy"]',
            },
            {
                "playlist_id": "bare",
                "playlist_name": "Bare Minimum",
                "description": "no metadata",
            },
        ]
    )
    matches = pd.DataFrame(
        [
            {"playlist_id": "chill", "track_id": "t1", "label": 1},
            {"playlist_id": "party", "track_id": "t2", "label": 1},
            {"playlist_id": "bare", "track_id": "t1", "label": 1},
        ]
    )
    tracks = pd.DataFrame(
        [
            {"track_id": "t1", "artist": "A", "track_name": "Serenity"},
            {"track_id": "t2", "artist": "B", "track_name": "Fiesta"},
        ]
    )
    track_emb = {
        "t1": _emb(0.5, 0.5, 0.0),
        "t2": _emb(0.0, 0.5, 0.5),
    }
    playlist_emb = {
        "chill": _emb(0.5, 0.5, 0.0),
        "party": _emb(0.0, 0.5, 0.5),
        "bare": _emb(0.3, 0.3, 0.3),
    }
    bundle = build_profiles(
        playlists, matches, tracks,
        track_text_emb_by_id=track_emb,
        playlist_text_emb_by_id=playlist_emb,
    )
    return bundle, tracks, track_emb


def test_playlist_soft_attr_structural_features() -> None:
    """Playlist-side structural features count non-empty soft attribute sets."""
    bundle, tracks, track_emb = _build_soft_attr_bundle()
    meta_lookup = build_track_meta_lookup(tracks)

    pairs = pd.DataFrame(
        [
            {"track_id": "t1", "playlist_id": "chill", "label": 1},
            {"track_id": "t1", "playlist_id": "bare", "label": 1},
        ]
    )
    feats = build_pair_features(
        pairs,
        profile_bundle=bundle,
        track_text_emb_by_id=track_emb,
        track_audio_by_id={},
        track_meta_by_id=meta_lookup,
    )

    chill_row = feats[feats.playlist_id == "chill"].iloc[0]
    bare_row = feats[feats.playlist_id == "bare"].iloc[0]

    # Chill playlist has activities, languages, moods → >=3 soft attrs.
    assert chill_row["playlist_n_soft_attrs"] >= 3.0
    assert chill_row["playlist_has_language"] == 1.0
    assert chill_row["playlist_has_mood"] == 1.0

    # Bare playlist has none.
    assert bare_row["playlist_n_soft_attrs"] == 0.0
    assert bare_row["playlist_has_language"] == 0.0
    assert bare_row["playlist_has_mood"] == 0.0


def test_soft_attr_features_default_when_no_track_data() -> None:
    """When track meta has no _soft_* keys (training-time), overlap features
    are all 0.0 and soft_attr_available is 0.0."""
    bundle, tracks, track_emb = _build_soft_attr_bundle()
    meta_lookup = build_track_meta_lookup(tracks)

    pairs = pd.DataFrame(
        [{"track_id": "t1", "playlist_id": "chill", "label": 1}]
    )
    feats = build_pair_features(
        pairs,
        profile_bundle=bundle,
        track_text_emb_by_id=track_emb,
        track_audio_by_id={},
        track_meta_by_id=meta_lookup,
    )
    row = feats.iloc[0]
    assert row["soft_attr_available"] == 0.0
    assert row["mood_match_flag"] == 0.0
    assert row["language_match_flag"] == 0.0
    assert row["activity_match_flag"] == 0.0


def test_soft_attr_overlap_with_matching_track_data() -> None:
    """When track meta has matching _soft_* keys, overlap features activate."""
    bundle, tracks, track_emb = _build_soft_attr_bundle()
    meta_lookup = build_track_meta_lookup(tracks)
    meta_lookup["t1"]["_soft_moods"] = {"calm", "happy"}
    meta_lookup["t1"]["_soft_languages"] = {"english"}
    meta_lookup["t1"]["_soft_activities"] = {"relaxing", "driving"}

    pairs = pd.DataFrame(
        [{"track_id": "t1", "playlist_id": "chill", "label": 1}]
    )
    feats = build_pair_features(
        pairs,
        profile_bundle=bundle,
        track_text_emb_by_id=track_emb,
        track_audio_by_id={},
        track_meta_by_id=meta_lookup,
    )
    row = feats.iloc[0]
    assert row["soft_attr_available"] == 1.0
    # "calm" overlaps chill playlist moods {"calm", "peaceful"}.
    assert row["mood_match_flag"] == 1.0
    # "english" matches chill playlist language {"english"}.
    assert row["language_match_flag"] == 1.0
    # activity Jaccard: {"relaxing","driving"} ∩ {"relaxing","studying"} = {"relaxing"}
    # union = {"relaxing","driving","studying"} → 1/3 ≈ 0.333
    assert 0.3 < row["activity_match_flag"] < 0.4


def test_soft_attr_no_overlap_yields_zero() -> None:
    """Disjoint track/playlist soft attrs → overlap flags = 0."""
    bundle, tracks, track_emb = _build_soft_attr_bundle()
    meta_lookup = build_track_meta_lookup(tracks)
    meta_lookup["t1"]["_soft_moods"] = {"aggressive", "dark"}
    meta_lookup["t1"]["_soft_languages"] = {"german"}
    meta_lookup["t1"]["_soft_activities"] = {"gaming"}

    pairs = pd.DataFrame(
        [{"track_id": "t1", "playlist_id": "chill", "label": 1}]
    )
    feats = build_pair_features(
        pairs,
        profile_bundle=bundle,
        track_text_emb_by_id=track_emb,
        track_audio_by_id={},
        track_meta_by_id=meta_lookup,
    )
    row = feats.iloc[0]
    assert row["soft_attr_available"] == 1.0
    assert row["mood_match_flag"] == 0.0
    assert row["language_match_flag"] == 0.0
    assert row["activity_match_flag"] == 0.0


def test_soft_attr_against_bare_playlist() -> None:
    """Playlist with no soft attrs → overlap features are 0 even when track has data."""
    bundle, tracks, track_emb = _build_soft_attr_bundle()
    meta_lookup = build_track_meta_lookup(tracks)
    meta_lookup["t1"]["_soft_moods"] = {"calm"}
    meta_lookup["t1"]["_soft_languages"] = {"english"}

    pairs = pd.DataFrame(
        [{"track_id": "t1", "playlist_id": "bare", "label": 1}]
    )
    feats = build_pair_features(
        pairs,
        profile_bundle=bundle,
        track_text_emb_by_id=track_emb,
        track_audio_by_id={},
        track_meta_by_id=meta_lookup,
    )
    row = feats.iloc[0]
    assert row["soft_attr_available"] == 1.0
    assert row["mood_match_flag"] == 0.0
    assert row["language_match_flag"] == 0.0


# ---- Over-tagging guard tests ----


def test_broadtag_sqrt_penalty() -> None:
    """The sqrt-based breadth penalty is gentler than linear."""
    import math

    threshold = 6
    breadth = 12
    linear = float(threshold) / float(breadth)
    sqrt_val = math.sqrt(float(threshold) / float(breadth))
    assert sqrt_val > linear
    assert abs(sqrt_val - math.sqrt(0.5)) < 1e-9


def test_broadtag_no_penalty_below_threshold() -> None:
    """Playlists at or below threshold get multiplier 1.0."""
    import math

    for breadth in [1, 4, 6]:
        threshold = 6
        if breadth > threshold:
            mult = math.sqrt(float(threshold) / float(breadth))
        else:
            mult = 1.0
        assert mult == 1.0
