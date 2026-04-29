from __future__ import annotations

import numpy as np
import pandas as pd

from matcher_agent.features.playlist_profiles import build_profiles


def _emb(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def test_build_profiles_uses_accepted_tracks_for_centroid() -> None:
    playlists = pd.DataFrame(
        [
            {"playlist_id": "p1", "playlist_name": "Hip-Hop Hits", "description": "rap and trap"},
        ]
    )
    matches = pd.DataFrame(
        [
            {"playlist_id": "p1", "track_id": "t1", "label": 1},
            {"playlist_id": "p1", "track_id": "t2", "label": 0},
        ]
    )
    tracks = pd.DataFrame(
        [
            {"track_id": "t1", "artist": "A", "track_name": "Foo", "bpm": 100, "loudness": -8},
            {"track_id": "t2", "artist": "B", "track_name": "Bar", "bpm": 80, "loudness": -10},
        ]
    )

    track_emb = {
        "t1": _emb(1.0, 0.0, 0.0),
        "t2": _emb(0.0, 1.0, 0.0),
    }
    playlist_emb = {"p1": _emb(0.0, 0.0, 1.0)}

    bundle = build_profiles(
        playlists,
        matches,
        tracks,
        track_text_emb_by_id=track_emb,
        playlist_text_emb_by_id=playlist_emb,
        semantic_blend=0.5,
    )

    prof = bundle.profiles["p1"]
    # Semantic centroid mixes playlist text emb (1, 0, 0 dim 3) and the
    # accepted-track centroid (just t1 → (1,0,0)).
    assert prof.semantic_centroid.shape == (3,)
    assert prof.tags == {"hip_hop"}
    assert prof.accepted_count == 1
    assert prof.declined_count == 1
    assert prof.acceptance_rate == 0.5
    assert prof.audio_centroid is not None
    assert prof.audio_centroid[0] == 100.0  # bpm centroid uses only accepted track t1


def test_build_profiles_handles_playlist_without_history() -> None:
    playlists = pd.DataFrame(
        [{"playlist_id": "p1", "playlist_name": "Country", "description": ""}]
    )
    matches = pd.DataFrame(columns=["playlist_id", "track_id", "label"])
    tracks = pd.DataFrame(columns=["track_id", "artist", "track_name"])

    bundle = build_profiles(
        playlists,
        matches,
        tracks,
        track_text_emb_by_id={},
        playlist_text_emb_by_id={"p1": _emb(0.0, 1.0, 0.0)},
    )
    prof = bundle.profiles["p1"]
    assert prof.accepted_count == 0
    assert prof.audio_centroid is None
    assert prof.tags == {"country"}
    # Falls back to playlist text embedding (then normalized)
    assert prof.semantic_centroid.shape == (3,)


def test_build_profiles_uses_xano_genres_and_subgenres() -> None:
    # Playlist text is intentionally generic ("daily mix") so the only way
    # we get specific tags is via the Xano genres/subgenres arrays.
    playlists = pd.DataFrame(
        [
            {
                "playlist_id": "p1",
                "playlist_name": "Daily Mix",
                "description": "fresh updates",
                "genres": ["Hip-Hop"],
                "subgenres": ["Trap", "Drill", "Lo-fi Hip-Hop"],
            }
        ]
    )
    matches = pd.DataFrame(columns=["playlist_id", "track_id", "label"])
    tracks = pd.DataFrame(columns=["track_id", "artist", "track_name"])

    bundle = build_profiles(
        playlists,
        matches,
        tracks,
        track_text_emb_by_id={},
        playlist_text_emb_by_id={"p1": _emb(1.0, 0.0)},
    )
    prof = bundle.profiles["p1"]
    assert "hip_hop" in prof.tags
    # "Lo-fi Hip-Hop" subgenre adds chill_lofi too.
    assert "chill_lofi" in prof.tags


def test_build_profiles_aggregates_popularity_from_accepted_tracks() -> None:
    playlists = pd.DataFrame(
        [{"playlist_id": "p1", "playlist_name": "Underground Indie", "description": ""}]
    )
    matches = pd.DataFrame(
        [
            {"playlist_id": "p1", "track_id": "t1", "label": 1},
            {"playlist_id": "p1", "track_id": "t2", "label": 1},
            # Declined: must NOT contribute to popularity stats.
            {"playlist_id": "p1", "track_id": "t3", "label": 0},
        ]
    )
    tracks = pd.DataFrame(
        [
            {"track_id": "t1", "artist": "A", "track_name": "x", "popularity": 20},
            {"track_id": "t2", "artist": "B", "track_name": "y", "popularity": 30},
            {"track_id": "t3", "artist": "C", "track_name": "z", "popularity": 90},
        ]
    )

    bundle = build_profiles(
        playlists,
        matches,
        tracks,
        track_text_emb_by_id={
            "t1": _emb(1.0, 0.0),
            "t2": _emb(0.0, 1.0),
            "t3": _emb(0.5, 0.5),
        },
        playlist_text_emb_by_id={"p1": _emb(0.0, 1.0)},
    )
    prof = bundle.profiles["p1"]
    assert prof.popularity_count == 2
    assert prof.popularity_mean == 25.0
    assert prof.popularity_std is not None
    assert prof.popularity_std > 0


def test_build_profiles_no_popularity_for_playlist_without_accepted_pop_data() -> None:
    playlists = pd.DataFrame(
        [{"playlist_id": "p1", "playlist_name": "Empty", "description": ""}]
    )
    matches = pd.DataFrame(
        [{"playlist_id": "p1", "track_id": "t1", "label": 1}]
    )
    tracks = pd.DataFrame(
        [{"track_id": "t1", "artist": "A", "track_name": "x"}]  # no popularity column
    )

    bundle = build_profiles(
        playlists,
        matches,
        tracks,
        track_text_emb_by_id={"t1": _emb(1.0, 0.0)},
        playlist_text_emb_by_id={"p1": _emb(0.0, 1.0)},
    )
    prof = bundle.profiles["p1"]
    assert prof.popularity_mean is None
    assert prof.popularity_std is None
    assert prof.popularity_count == 0
