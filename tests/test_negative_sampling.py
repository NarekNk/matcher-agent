from __future__ import annotations

import numpy as np
import pandas as pd

from matcher_agent.training.dataset import build_training_bundle
from matcher_agent.training.train_ranker import _augment_with_random_negatives


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
