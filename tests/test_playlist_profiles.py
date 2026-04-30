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


def test_build_profiles_separates_primary_tags_from_subgenre_tags() -> None:
    """`primary_tags` must hold canonical tags coming exclusively from the
    Xano top-level `genres` array. `tags` is the union of primary + subgenre
    + text-derived tags. The strict explicit-genre filter relies on this
    distinction to demote 'Rock playlist with Blues Rock subgenre' below a
    real Blues primary playlist."""
    playlists = pd.DataFrame(
        [
            {
                "playlist_id": "p_rock_with_bluesrock",
                "playlist_name": "Made for You",
                "description": "guitar-heavy",
                "genres": ["Rock", "Acoustic"],
                "subgenres": ["Blues Rock", "Hard Rock", "Folk Rock"],
            },
            {
                "playlist_id": "p_blues_primary",
                "playlist_name": "Real Blues",
                "description": "delta + chicago",
                "genres": ["Blues"],
                "subgenres": ["Delta Blues", "Chicago Blues"],
            },
        ]
    )
    matches = pd.DataFrame(columns=["playlist_id", "track_id", "label"])
    tracks = pd.DataFrame(columns=["track_id", "artist", "track_name"])

    bundle = build_profiles(
        playlists,
        matches,
        tracks,
        track_text_emb_by_id={},
        playlist_text_emb_by_id={
            "p_rock_with_bluesrock": _emb(1.0, 0.0),
            "p_blues_primary": _emb(0.0, 1.0),
        },
    )
    rock = bundle.profiles["p_rock_with_bluesrock"]
    blues = bundle.profiles["p_blues_primary"]

    # Rock playlist's PRIMARY tags must NOT include `blues` even though
    # `Blues Rock` (a subgenre) maps to {rock, blues}. `blues` must only
    # show up in the union `tags`.
    assert "blues" not in rock.primary_tags
    assert rock.primary_tags == {"rock", "folk_acoustic"}
    assert "blues" in rock.tags
    # Real Blues playlist's primary set IS blues.
    assert blues.primary_tags == {"blues"}


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


def test_build_profiles_normalizes_soft_attributes() -> None:
    playlists = pd.DataFrame(
        [
            {
                "playlist_id": "p1",
                "playlist_name": "Workout Hits",
                "description": "high energy",
                # Note: "Any" / "Other" / null must be filtered out.
                "activity": ["Workout", "any"],
                "countries": ["Other"],  # only no-signal value -> empty
                "languages": ["English"],
                "tempos": [],
                "moods": ["Energetic", "Uplifting"],
            },
            {
                "playlist_id": "p2",
                "playlist_name": "Sleep Time",
                "description": "calm",
                "activity": ["Relax"],
                "countries": [],
                "languages": ["english"],
                "tempos": ["Slow"],
                "moods": ["Calm"],
            },
        ]
    )
    matches = pd.DataFrame(columns=["playlist_id", "track_id", "label"])
    tracks = pd.DataFrame(columns=["track_id", "artist", "track_name"])

    bundle = build_profiles(
        playlists,
        matches,
        tracks,
        track_text_emb_by_id={},
        playlist_text_emb_by_id={"p1": _emb(1.0, 0.0), "p2": _emb(0.0, 1.0)},
    )
    p1 = bundle.profiles["p1"]
    p2 = bundle.profiles["p2"]
    assert p1.activities == {"workout"}
    assert p1.countries == set()
    assert p1.languages == {"english"}
    assert p1.tempos == set()
    assert p1.moods == {"energetic", "uplifting"}

    assert p2.activities == {"relax"}
    assert p2.languages == {"english"}
    assert p2.tempos == {"slow"}
    assert p2.moods == {"calm"}

    # Soft attributes should also be exposed via the dict accessor used
    # by the soft-penalty multiplier in the inference service.
    soft = p1.soft_attribute_sets()
    assert soft["activities"] == {"workout"}
    assert soft["moods"] == {"energetic", "uplifting"}


def test_build_profiles_handles_missing_soft_attribute_columns() -> None:
    """Older playlist parquet files won't have the new attribute columns.
    The profile builder must default to empty sets without crashing."""
    playlists = pd.DataFrame(
        [{"playlist_id": "p1", "playlist_name": "Old Playlist", "description": ""}]
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
    assert prof.activities == set()
    assert prof.countries == set()
    assert prof.languages == set()
    assert prof.tempos == set()
    assert prof.moods == set()
