from __future__ import annotations

from matcher_agent.features.genre_normalizer import (
    normalize_external_labels,
    normalize_xano_labels,
)


def test_normalize_xano_labels_maps_top_level_genres() -> None:
    tags = normalize_xano_labels(["Hip-Hop"], None)
    assert tags == {"hip_hop"}

    tags = normalize_xano_labels(["Pop", "Rock"], None)
    assert tags == {"pop", "rock"}


def test_normalize_xano_labels_ignores_other_and_empty() -> None:
    assert normalize_xano_labels(None, None) == set()
    assert normalize_xano_labels(["Other"], []) == set()


def test_normalize_xano_labels_picks_up_subgenres_with_multiple_tags() -> None:
    # The Xano example: Hip-Hop / Trap / Lo-fi Hip-Hop -> hip_hop + chill_lofi
    tags = normalize_xano_labels(["Hip-Hop"], ["Trap", "Lo-fi Hip-Hop"])
    assert "hip_hop" in tags
    assert "chill_lofi" in tags


def test_normalize_xano_labels_metal_is_not_just_rock() -> None:
    # Metal must produce its own canonical tag so the conflict detector
    # can keep metal away from pop/lofi/etc.
    tags = normalize_xano_labels(["Metal"], ["Heavy Metal", "Doom Metal"])
    assert tags == {"metal"}


def test_normalize_xano_labels_punk_is_distinct_from_rock() -> None:
    tags = normalize_xano_labels(["Punk"], ["Pop Punk", "Hardcore Punk"])
    assert "punk" in tags
    assert "pop" in tags  # Pop Punk is multi-tag
    assert "rock" not in tags


def test_normalize_xano_pop_rock_maps_to_compound_not_standalone_rock() -> None:
    tags = normalize_xano_labels(["Pop"], ["Pop Rock", "Teen Pop"])
    assert "pop_rock" in tags
    assert "pop" in tags
    assert "rock" not in tags


def test_normalize_xano_dance_pop_is_pop_not_workout_party() -> None:
    tags = normalize_xano_labels(["Pop"], ["Dance Pop"])
    assert "pop" in tags
    assert "workout_party" not in tags


def test_normalize_external_labels_handles_spotify_strings() -> None:
    # Spotify artist genre strings may match either an explicit subgenre
    # entry or fall back to the regex tagger.
    tags = normalize_external_labels(["west coast rap", "trap", "synth pop"])
    assert "hip_hop" in tags
    assert "pop" in tags
    assert "edm" in tags


def test_normalize_external_labels_with_unknown_label_falls_back_to_regex() -> None:
    # "modern country" isn't in our explicit subgenre map but the regex
    # tagger picks up "country".
    tags = normalize_external_labels(["modern country"])
    assert tags == {"country"}


def test_normalize_external_labels_empty_returns_empty_set() -> None:
    assert normalize_external_labels(None) == set()
    assert normalize_external_labels([]) == set()
    assert normalize_external_labels(["", "   "]) == set()
