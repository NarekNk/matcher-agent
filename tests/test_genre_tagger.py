from __future__ import annotations

from matcher_agent.features.genre_tagger import (
    has_conflict,
    jaccard,
    tag_text,
)


def test_tag_text_recognizes_hip_hop_variants() -> None:
    assert tag_text("hip-hop rap urban daily") == {"hip_hop"}
    assert tag_text("HipHop & Trap") == {"hip_hop"}
    assert tag_text("Hip Hop / Drill 2026") == {"hip_hop"}


def test_tag_text_recognizes_country() -> None:
    assert tag_text("Country Music Summer 2026") == {"country"}
    assert tag_text("Bluegrass Revival") == {"country"}


def test_tag_text_recognizes_edm_keywords() -> None:
    assert "edm" in tag_text("Tropical House Vibes")
    assert "edm" in tag_text("Best Electro Music")
    assert "edm" in tag_text("Drum and Bass weekly")


def test_tag_text_can_return_multiple_genres() -> None:
    tags = tag_text("Indie Rock & Alt Folk Acoustic")
    assert "alt_indie" in tags
    assert "rock" in tags
    assert "folk_acoustic" in tags


def test_jaccard_basic() -> None:
    assert jaccard(set(), {"a"}) == 0.0
    assert jaccard({"a"}, {"a"}) == 1.0
    assert jaccard({"a", "b"}, {"a", "c"}) == 1 / 3


def test_has_conflict_flags_obvious_mismatches() -> None:
    track_tags = {"hip_hop"}
    playlist_tags = {"country"}
    assert has_conflict(track_tags, playlist_tags) is True


def test_has_conflict_no_flag_when_overlap() -> None:
    assert has_conflict({"hip_hop", "rnb"}, {"hip_hop"}) is False


def test_has_conflict_no_flag_when_unknown() -> None:
    assert has_conflict(set(), {"hip_hop"}) is False
