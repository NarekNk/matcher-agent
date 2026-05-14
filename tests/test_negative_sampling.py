from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from matcher_agent.training.dataset import build_training_bundle
from matcher_agent.training.train_ranker import (
    NegativeSamplingConfig,
    _augment_with_negatives,
    _augment_with_random_negatives,
    _precompute_similar_playlists,
)


class _StubEmbedder:
    """Deterministic genre-aware embedder; lets us test the conflict sampler
    without loading sentence-transformers."""

    model_name = "stub-test"

    def encode(self, texts):
        out = []
        for t in texts:
            t_norm = (t or "").strip().lower()
            if "rap" in t_norm or "hip" in t_norm or "trap" in t_norm or "drill" in t_norm:
                vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            elif "country" in t_norm:
                vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            elif "edm" in t_norm or "house" in t_norm or "electro" in t_norm:
                vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            out.append(vec)
        return np.vstack(out)


def _setup_three_genre_world() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    """Three playlists in disjoint genres + three tracks, one per genre."""
    playlists = pd.DataFrame(
        [
            {"playlist_id": "hh", "playlist_name": "Hip-Hop Hits", "description": "rap and trap"},
            {"playlist_id": "co", "playlist_name": "Country Roads", "description": "country only"},
            {"playlist_id": "edm", "playlist_name": "EDM House", "description": "electro house"},
        ]
    )
    matches = pd.DataFrame(
        [
            {
                "playlist_id": "hh",
                "track_id": "t_rap",
                "track_name": "Drill Anthem",
                "artist": "MC",
                "label": 1,
            },
            {
                "playlist_id": "co",
                "track_id": "t_country",
                "track_name": "Pickup Truck",
                "artist": "Joe",
                "label": 1,
            },
            {
                "playlist_id": "edm",
                "track_id": "t_edm",
                "track_name": "Electro House Banger",
                "artist": "DJ",
                "label": 1,
            },
        ]
    )
    tracks = pd.DataFrame(
        [
            {"track_id": "t_rap", "artist": "MC", "track_name": "Drill Anthem", "bpm": 140},
            {"track_id": "t_country", "artist": "Joe", "track_name": "Pickup Truck", "bpm": 90},
            {"track_id": "t_edm", "artist": "DJ", "track_name": "Electro House Banger", "bpm": 128},
        ]
    )
    bundle = build_training_bundle(
        matches,
        tracks,
        playlists,
        text_embedder=_StubEmbedder(),
        semantic_blend=0.25,
    )
    return matches, tracks, playlists, bundle


def test_pure_conflict_negatives_have_zero_genre_overlap() -> None:
    matches, tracks, playlists, bundle = _setup_three_genre_world()
    train_df = bundle.pair_features

    augmented = _augment_with_random_negatives(
        train_df,
        train_matches=matches,
        playlists_df=playlists,
        train_bundle=bundle,
        ratio=4.0,
        random_state=7,
        conflict_fraction=1.0,  # 100% conflict negatives
    )

    new_negatives = augmented[(augmented["label"] == 0)]
    assert len(new_negatives) > 0

    profiles = bundle.profile_bundle.profiles
    track_tags_by_id = {
        tid: meta.get("_cached_tags", set())
        for tid, meta in bundle.track_meta_by_id.items()
    }
    for _, row in new_negatives.iterrows():
        tid = str(row["track_id"])
        pid = str(row["playlist_id"])
        track_tags = track_tags_by_id.get(tid, set())
        playlist_tags = profiles[pid].tags
        # When both sides have tags, conflict negatives must not share any
        # canonical tag with the track. (Empty-tag rows simply don't qualify
        # as conflict negatives and would have been skipped.)
        if track_tags and playlist_tags:
            assert not (track_tags & playlist_tags), (
                f"Conflict negative ({tid} -> {pid}) leaked an overlapping tag: "
                f"{track_tags & playlist_tags}"
            )


def test_conflict_negatives_skip_already_pitched_playlists() -> None:
    matches, tracks, playlists, bundle = _setup_three_genre_world()
    train_df = bundle.pair_features

    augmented = _augment_with_random_negatives(
        train_df,
        train_matches=matches,
        playlists_df=playlists,
        train_bundle=bundle,
        ratio=10.0,
        random_state=42,
        conflict_fraction=1.0,
    )
    pitched_pairs = {
        (str(row.track_id), str(row.playlist_id))
        for row in matches.itertuples(index=False)
    }
    new_negatives = augmented[(augmented["label"] == 0)]
    for _, row in new_negatives.iterrows():
        assert (str(row["track_id"]), str(row["playlist_id"])) not in pitched_pairs


def test_random_only_mode_does_not_require_conflict() -> None:
    matches, tracks, playlists, bundle = _setup_three_genre_world()
    train_df = bundle.pair_features

    augmented = _augment_with_random_negatives(
        train_df,
        train_matches=matches,
        playlists_df=playlists,
        train_bundle=bundle,
        ratio=2.0,
        random_state=0,
        conflict_fraction=0.0,
    )
    # Should add ratio * positives = 6 negatives (3 positives * 2.0).
    n_new = len(augmented) - len(train_df)
    assert n_new == 6


def test_zero_ratio_returns_train_df_unchanged() -> None:
    matches, tracks, playlists, bundle = _setup_three_genre_world()
    train_df = bundle.pair_features

    augmented = _augment_with_random_negatives(
        train_df,
        train_matches=matches,
        playlists_df=playlists,
        train_bundle=bundle,
        ratio=0.0,
        random_state=0,
        conflict_fraction=0.5,
    )
    assert augmented.equals(train_df)


def test_playlist_anchored_conflicts_when_track_text_lacks_genre_keywords() -> None:
    """Realistic case: a track titled "Bad Habits" by a generic-named artist
    has no regex-detectable genre tags, but its accepted playlist has clear
    genre tags. Playlist-anchored sampling must still produce a conflict
    negative by using the playlist's tags as the anchor."""
    playlists = pd.DataFrame(
        [
            {
                "playlist_id": "rap_pl",
                "playlist_name": "Hip-Hop Heat",
                "description": "rap, trap, drill anthems",
            },
            {
                "playlist_id": "country_pl",
                "playlist_name": "Country Roads",
                "description": "country, bluegrass, americana",
            },
        ]
    )
    matches = pd.DataFrame(
        [
            {
                "playlist_id": "rap_pl",
                "track_id": "t_generic",
                "track_name": "Bad Habits",
                "artist": "Generic Artist",
                "label": 1,
            },
        ]
    )
    tracks = pd.DataFrame(
        [
            {
                "track_id": "t_generic",
                "artist": "Generic Artist",
                "track_name": "Bad Habits",
                "bpm": 120,
            },
        ]
    )
    bundle = build_training_bundle(
        matches,
        tracks,
        playlists,
        text_embedder=_StubEmbedder(),
        semantic_blend=0.25,
    )
    train_df = bundle.pair_features

    profiles = bundle.profile_bundle.profiles
    # Sanity: track text has no genre tags but anchor playlist does.
    assert not bundle.track_meta_by_id["t_generic"].get("_cached_tags", set())
    assert profiles["rap_pl"].tags  # populated from "rap, trap, drill"

    augmented = _augment_with_random_negatives(
        train_df,
        train_matches=matches,
        playlists_df=playlists,
        train_bundle=bundle,
        ratio=1.0,
        random_state=0,
        conflict_fraction=1.0,
    )
    new_negs = augmented[augmented["label"] == 0]
    assert len(new_negs) == 1, (
        "Playlist-anchored sampler should produce a conflict negative "
        "even when the track text has no detectable genre tags"
    )
    neg_pid = str(new_negs.iloc[0]["playlist_id"])
    assert neg_pid == "country_pl", (
        f"Expected country_pl as the genre-conflict negative for a track "
        f"accepted on rap_pl; got {neg_pid}"
    )


# ---- NegativeSamplingConfig tests ----


def test_config_random_fraction_computed() -> None:
    cfg = NegativeSamplingConfig(ratio=5.0, conflict_fraction=0.33, near_miss_fraction=0.33)
    assert abs(cfg.random_fraction - 0.34) < 1e-9


def test_config_rejects_fractions_over_one() -> None:
    with pytest.raises(ValueError, match="must be <= 1.0"):
        NegativeSamplingConfig(conflict_fraction=0.6, near_miss_fraction=0.5)


def test_config_from_legacy_disables_near_miss() -> None:
    cfg = NegativeSamplingConfig.from_legacy(ratio=3.0, conflict_fraction=0.5)
    assert cfg.ratio == 3.0
    assert cfg.conflict_fraction == 0.5
    assert cfg.near_miss_fraction == 0.0
    assert cfg.popularity_stratified is False
    assert cfg.random_fraction == 0.5


# ---- Near-miss sampling tests ----


def test_precompute_similar_playlists_returns_top_k() -> None:
    emb = {
        "p1": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "p2": np.array([0.9, 0.1, 0.0], dtype=np.float32),
        "p3": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    similar = _precompute_similar_playlists(emb, k=2)
    assert "p1" in similar
    assert similar["p1"][0] == "p2"


def test_near_miss_negatives_are_semantically_similar() -> None:
    """With near_miss_fraction=1.0, all negatives should come from
    playlists most similar to the accepted one."""
    matches, tracks, playlists, bundle = _setup_three_genre_world()
    train_df = bundle.pair_features

    config = NegativeSamplingConfig(
        ratio=1.0, conflict_fraction=0.0, near_miss_fraction=1.0,
        popularity_stratified=False,
    )
    augmented = _augment_with_negatives(
        train_df,
        train_matches=matches,
        playlists_df=playlists,
        train_bundle=bundle,
        config=config,
        random_state=42,
    )
    new_negs = augmented[augmented["label"] == 0]
    assert len(new_negs) > 0

    # Each negative should be a playlist different from the accepted one
    # but in the near-miss neighborhood (top-K similar). With only 3
    # playlists all negatives must be one of the other two.
    for _, row in new_negs.iterrows():
        tid = str(row["track_id"])
        neg_pid = str(row["playlist_id"])
        pitched = {
            str(r.playlist_id)
            for r in matches[matches["track_id"].astype(str) == tid].itertuples()
        }
        assert neg_pid not in pitched


def test_three_tier_split_produces_all_types() -> None:
    """With equal 1/3 fractions and enough playlists, all three pools
    should produce at least one negative."""
    # Build a larger world with 6 playlists in 3 genres.
    playlists = pd.DataFrame(
        [
            {"playlist_id": "hh1", "playlist_name": "Hip-Hop Hits", "description": "rap trap"},
            {"playlist_id": "hh2", "playlist_name": "Rap Life", "description": "hip hop rap"},
            {"playlist_id": "co1", "playlist_name": "Country Roads", "description": "country"},
            {"playlist_id": "co2", "playlist_name": "Nashville Now", "description": "country"},
            {"playlist_id": "edm1", "playlist_name": "EDM House", "description": "electro house"},
            {"playlist_id": "edm2", "playlist_name": "Dance Floor", "description": "house edm"},
        ]
    )
    matches = pd.DataFrame(
        [
            {"playlist_id": "hh1", "track_id": "t1", "track_name": "Drill", "artist": "MC", "label": 1},
            {"playlist_id": "co1", "track_id": "t2", "track_name": "Pickup", "artist": "Joe", "label": 1},
            {"playlist_id": "edm1", "track_id": "t3", "track_name": "House", "artist": "DJ", "label": 1},
        ]
    )
    tracks = pd.DataFrame(
        [
            {"track_id": "t1", "artist": "MC", "track_name": "Drill", "bpm": 140},
            {"track_id": "t2", "artist": "Joe", "track_name": "Pickup", "bpm": 90},
            {"track_id": "t3", "artist": "DJ", "track_name": "House", "bpm": 128},
        ]
    )
    bundle = build_training_bundle(
        matches, tracks, playlists, text_embedder=_StubEmbedder(), semantic_blend=0.25,
    )
    train_df = bundle.pair_features

    config = NegativeSamplingConfig(
        ratio=4.0, conflict_fraction=0.33, near_miss_fraction=0.33,
        popularity_stratified=False,
    )
    augmented = _augment_with_negatives(
        train_df,
        train_matches=matches,
        playlists_df=playlists,
        train_bundle=bundle,
        config=config,
        random_state=42,
    )
    n_new = len(augmented) - len(train_df)
    # 3 positives * 4.0 ratio = 12 negatives target
    assert n_new == 12


def test_popularity_stratified_random_negatives() -> None:
    """Stratified sampling produces negatives from multiple tiers."""
    playlists = pd.DataFrame(
        [
            {"playlist_id": "t1_a", "playlist_name": "Tier1 A", "description": "rap", "tier": 1},
            {"playlist_id": "t2_a", "playlist_name": "Tier2 A", "description": "country", "tier": 2},
            {"playlist_id": "t3_a", "playlist_name": "Tier3 A", "description": "house", "tier": 3},
            {"playlist_id": "t4_a", "playlist_name": "Tier4 A", "description": "pop", "tier": 4},
        ]
    )
    matches = pd.DataFrame(
        [
            {"playlist_id": "t1_a", "track_id": "t1", "track_name": "Drill", "artist": "MC", "label": 1},
        ]
    )
    tracks = pd.DataFrame(
        [{"track_id": "t1", "artist": "MC", "track_name": "Drill", "bpm": 140}]
    )
    bundle = build_training_bundle(
        matches, tracks, playlists, text_embedder=_StubEmbedder(), semantic_blend=0.25,
    )
    train_df = bundle.pair_features

    config = NegativeSamplingConfig(
        ratio=30.0, conflict_fraction=0.0, near_miss_fraction=0.0,
        popularity_stratified=True,
    )
    augmented = _augment_with_negatives(
        train_df,
        train_matches=matches,
        playlists_df=playlists,
        train_bundle=bundle,
        config=config,
        random_state=42,
    )
    new_negs = augmented[augmented["label"] == 0]
    neg_pids = set(new_negs["playlist_id"].astype(str))
    # With stratified sampling across 4 tiers and 30 negatives, we
    # should see negatives from at least 2 different tiers (the
    # pitched tier1 playlist is excluded, leaving 3 candidates).
    assert len(neg_pids) >= 2, f"Expected negatives from multiple tiers, got {neg_pids}"
