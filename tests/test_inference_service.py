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
        if not texts:
            return np.empty((0, 3), dtype=np.float32)
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


def test_soft_attribute_penalty_demotes_conflicting_playlist(tmp_path: Path) -> None:
    """Two genre-equivalent playlists, one with mood=energetic and one with
    mood=calm. A track marked mood=calm should rank the calm playlist first
    even though their model scores are otherwise identical."""
    pipe = _make_dummy_pipeline(PAIRWISE_FEATURE_COLS)
    save_bundle({"model": pipe, "feature_columns": PAIRWISE_FEATURE_COLS}, tmp_path)

    historical = pd.DataFrame(columns=["track_id", "playlist_id", "label", "track_name", "artist"])
    playlists = pd.DataFrame(
        [
            {
                "playlist_id": "p_energetic",
                "playlist_name": "Hip-Hop Workout",
                "description": "trap workout",
                "moods": ["Energetic"],
                "activity": ["Workout"],
            },
            {
                "playlist_id": "p_calm",
                "playlist_name": "Hip-Hop Chill",
                "description": "trap chill",
                "moods": ["Calm"],
                "activity": ["Relax"],
            },
        ]
    )
    tracks = pd.DataFrame(columns=["track_id", "track_name", "artist", "bpm"])

    service = MatcherService(
        artifact_dir=str(tmp_path),
        historical_df=historical,
        playlists_df=playlists,
        tracks_df=tracks,
        text_embedder=_StubEmbedder(),
        hard_genre_filter=False,  # isolate the soft penalty path
        soft_attribute_penalty=0.4,
    )

    # First: no soft inputs from the user => both playlists tied, deterministic order.
    baseline = service.recommend_playlists(
        TrackInput(track_id="t1", track_name="Trap Banger", artist="Newcomer"), n=2
    )
    assert {r.playlist_id for r in baseline} == {"p_energetic", "p_calm"}

    # With user-supplied calm/relax: the energetic+workout playlist conflicts
    # on TWO attributes and gets penalized 0.4*0.4=0.16x.
    biased = service.recommend_playlists(
        TrackInput(
            track_id="t1",
            track_name="Trap Banger",
            artist="Newcomer",
            moods=["calm"],
            activities=["relax"],
        ),
        n=2,
    )
    assert biased[0].playlist_id == "p_calm"
    assert biased[1].playlist_id == "p_energetic"
    # The penalized score should be strictly lower than the matching one.
    assert biased[0].acceptance_probability > biased[1].acceptance_probability


def test_language_mismatch_penalty_demotes_wrong_language_playlist(tmp_path: Path) -> None:
    """Playlists tagged with a different language than the track should rank
    below same-genre playlists tagged with a matching language when the user
    supplies `--track-language`."""
    pipe = _make_dummy_pipeline(PAIRWISE_FEATURE_COLS)
    save_bundle({"model": pipe, "feature_columns": PAIRWISE_FEATURE_COLS}, tmp_path)

    historical = pd.DataFrame(columns=["track_id", "playlist_id", "label", "track_name", "artist"])
    playlists = pd.DataFrame(
        [
            {
                "playlist_id": "p_en",
                "playlist_name": "Playlist EN",
                # Same description on both rows so semantic features tie; only
                # language tags differ.
                "description": "rap drill hip hop",
                "languages": ["english"],
            },
            {
                "playlist_id": "p_pt",
                "playlist_name": "Playlist PT",
                "description": "rap drill hip hop",
                "languages": ["portuguese"],
            },
        ]
    )
    tracks = pd.DataFrame(columns=["track_id", "track_name", "artist", "bpm"])

    service = MatcherService(
        artifact_dir=str(tmp_path),
        historical_df=historical,
        playlists_df=playlists,
        tracks_df=tracks,
        text_embedder=_StubEmbedder(),
        hard_genre_filter=False,
        soft_attribute_penalty=1.0,
        language_mismatch_penalty=0.05,
    )
    recs = service.recommend_playlists(
        TrackInput(
            track_id="t1",
            track_name="Trap",
            artist="MC",
            languages=["english"],
        ),
        n=2,
    )
    assert recs[0].playlist_id == "p_en"
    assert recs[1].playlist_id == "p_pt"
    assert recs[0].acceptance_probability > recs[1].acceptance_probability


def test_track_text_includes_user_supplied_genres_in_embedding(tmp_path: Path) -> None:
    """Fix A: user-supplied genres/subgenres must reach the embedded track
    text so semantic_similarity (the dominant feature) becomes genre-aware
    even when Spotify has no `artist_genres` for the artist."""
    from matcher_agent.inference.service import MatcherService

    pipe = _make_dummy_pipeline(PAIRWISE_FEATURE_COLS)
    save_bundle({"model": pipe, "feature_columns": PAIRWISE_FEATURE_COLS}, tmp_path)
    historical = pd.DataFrame(columns=["track_id", "playlist_id", "label", "track_name", "artist"])
    playlists = pd.DataFrame(
        [{"playlist_id": "p1", "playlist_name": "Generic", "description": ""}]
    )
    tracks = pd.DataFrame(columns=["track_id", "track_name", "artist", "bpm"])

    service = MatcherService(
        artifact_dir=str(tmp_path),
        historical_df=historical,
        playlists_df=playlists,
        tracks_df=tracks,
        text_embedder=_StubEmbedder(),
        hard_genre_filter=False,
    )
    text_no_genres = service._track_text_for_input(
        TrackInput(track_id="t1", track_name="Gunpowder", artist="Sarah Dunn Music")
    )
    assert "genres:" not in text_no_genres.lower()

    text_user_genres = service._track_text_for_input(
        TrackInput(
            track_id="t1",
            track_name="Gunpowder",
            artist="Sarah Dunn Music",
            genres=["Blues"],
            subgenres=["Soul", "Jazz"],
        )
    )
    # All three user-supplied terms must show up, lowercased, in the
    # injected `Genres: ...` phrase so the sentence transformer can carry
    # them into semantic_similarity.
    lowered = text_user_genres.lower()
    assert "genres:" in lowered
    assert "blues" in lowered
    assert "soul" in lowered
    assert "jazz" in lowered

    text_combined = service._track_text_for_input(
        TrackInput(
            track_id="t1",
            track_name="Gunpowder",
            artist="Sarah Dunn Music",
            artist_genres=["singer-songwriter"],
            genres=["Blues"],
            subgenres=["Soul"],
        )
    )
    lowered_c = text_combined.lower()
    assert "singer-songwriter" in lowered_c
    assert "blues" in lowered_c
    assert "soul" in lowered_c

    text_lang = service._track_text_for_input(
        TrackInput(
            track_id="t1",
            track_name="Gunpowder",
            artist="Sarah Dunn Music",
            languages=["English"],
        )
    )
    assert "language:" in text_lang.lower()
    assert "english" in text_lang.lower()


def test_explicit_genre_filter_drops_non_overlapping_playlists(tmp_path: Path) -> None:
    """Fix B: when the user explicitly supplies track genres, playlists whose
    Xano tags don't overlap must be heavily penalized regardless of whether
    the conflict groups would have caught them."""
    pipe = _make_dummy_pipeline(PAIRWISE_FEATURE_COLS)
    save_bundle({"model": pipe, "feature_columns": PAIRWISE_FEATURE_COLS}, tmp_path)
    historical = pd.DataFrame(columns=["track_id", "playlist_id", "label", "track_name", "artist"])
    # Three playlists with disjoint genre signals. None of these would be
    # caught by `_CONFLICT_GROUPS` against {blues, soul, jazz}:
    #   p_pop has only `pop` (no {pop, jazz}/{pop, blues} conflict registered)
    #   p_latin has only `latin` (no {latin, jazz}/{latin, blues} conflict)
    #   p_jazz has `jazz` -> shares with the user request
    playlists = pd.DataFrame(
        [
            {
                "playlist_id": "p_pop",
                "playlist_name": "Top Pop Hits",
                "description": "the biggest pop tracks",
                "genres": ["Pop"],
            },
            {
                "playlist_id": "p_latin",
                "playlist_name": "Latin Verao",
                "description": "tropical hits",
                "genres": ["Latin"],
            },
            {
                "playlist_id": "p_jazz",
                "playlist_name": "Smooth Jazz",
                "description": "jazz selections",
                "genres": ["Jazz"],
            },
        ]
    )
    tracks = pd.DataFrame(columns=["track_id", "track_name", "artist", "bpm"])

    service = MatcherService(
        artifact_dir=str(tmp_path),
        historical_df=historical,
        playlists_df=playlists,
        tracks_df=tracks,
        text_embedder=_StubEmbedder(),
        hard_genre_filter=True,
        # Use a clearly-distinguishable penalty so the assertion is robust to
        # any small fluctuations in the dummy model's raw scores.
        explicit_genre_no_match_penalty=0.01,
        explicit_genre_untagged_penalty=0.3,
    )
    recs = service.recommend_playlists(
        TrackInput(
            track_id="t_blues",
            track_name="Gunpowder",
            artist="Sarah Dunn Music",
            genres=["Blues"],
            subgenres=["Soul", "Jazz"],
        ),
        n=3,
    )
    # The only overlapping playlist must rank #1.
    assert recs[0].playlist_id == "p_jazz"
    # And the non-overlapping playlists must be dropped well below the jazz one.
    jazz_score = next(r.acceptance_probability for r in recs if r.playlist_id == "p_jazz")
    for rec in recs:
        if rec.playlist_id != "p_jazz":
            assert rec.acceptance_probability <= jazz_score * 0.05, (
                f"expected non-overlap playlist {rec.playlist_id} to be heavily "
                f"penalized vs jazz; got {rec.acceptance_probability} vs {jazz_score}"
            )


def test_explicit_genre_filter_softly_penalizes_untagged_playlists(tmp_path: Path) -> None:
    """Fix C: untagged playlists (no Xano genres, generic name) are softly
    penalized when the user supplies explicit genres. They should rank below
    overlapping playlists but ABOVE non-overlapping tagged playlists."""
    pipe = _make_dummy_pipeline(PAIRWISE_FEATURE_COLS)
    save_bundle({"model": pipe, "feature_columns": PAIRWISE_FEATURE_COLS}, tmp_path)
    historical = pd.DataFrame(columns=["track_id", "playlist_id", "label", "track_name", "artist"])
    playlists = pd.DataFrame(
        [
            {
                "playlist_id": "p_jazz",
                "playlist_name": "Smooth Jazz",
                "description": "jazz",
                "genres": ["Jazz"],
            },
            {
                "playlist_id": "p_untagged",
                "playlist_name": "Daily Mix",
                "description": "fresh songs",
            },
            {
                "playlist_id": "p_pop",
                "playlist_name": "Pop Hits",
                "description": "pop",
                "genres": ["Pop"],
            },
        ]
    )
    tracks = pd.DataFrame(columns=["track_id", "track_name", "artist", "bpm"])

    service = MatcherService(
        artifact_dir=str(tmp_path),
        historical_df=historical,
        playlists_df=playlists,
        tracks_df=tracks,
        text_embedder=_StubEmbedder(),
        hard_genre_filter=True,
        explicit_genre_no_match_penalty=0.01,
        explicit_genre_untagged_penalty=0.3,
    )
    recs = service.recommend_playlists(
        TrackInput(
            track_id="t1",
            track_name="Gunpowder",
            artist="Sarah Dunn Music",
            genres=["Jazz"],
        ),
        n=3,
    )
    by_id = {r.playlist_id: r.acceptance_probability for r in recs}
    # Overlap > untagged > no-overlap.
    assert by_id["p_jazz"] >= by_id["p_untagged"]
    assert by_id["p_untagged"] > by_id["p_pop"]


def test_explicit_genre_filter_subgenre_only_match_is_moderately_penalized(tmp_path: Path) -> None:
    """Fix F: a Rock playlist whose ONLY blues signal is the 'Blues Rock'
    subgenre must rank below a real Blues primary playlist. The previous
    'any overlap = pass' rule treated them equally."""
    pipe = _make_dummy_pipeline(PAIRWISE_FEATURE_COLS)
    save_bundle({"model": pipe, "feature_columns": PAIRWISE_FEATURE_COLS}, tmp_path)
    historical = pd.DataFrame(columns=["track_id", "playlist_id", "label", "track_name", "artist"])
    playlists = pd.DataFrame(
        [
            {
                "playlist_id": "p_blues_primary",
                "playlist_name": "Real Blues",
                "description": "delta + chicago",
                "genres": ["Blues"],
                "subgenres": ["Delta Blues"],
            },
            {
                "playlist_id": "p_rock_with_bluesrock",
                "playlist_name": "Made for You",
                "description": "rock playlist",
                "genres": ["Rock"],
                "subgenres": ["Blues Rock", "Hard Rock"],
            },
        ]
    )
    tracks = pd.DataFrame(columns=["track_id", "track_name", "artist", "bpm"])

    service = MatcherService(
        artifact_dir=str(tmp_path),
        historical_df=historical,
        playlists_df=playlists,
        tracks_df=tracks,
        text_embedder=_StubEmbedder(),
        hard_genre_filter=True,
        explicit_genre_no_match_penalty=0.01,
        explicit_genre_untagged_penalty=0.3,
        explicit_genre_subgenre_only_penalty=0.4,
        explicit_genre_broadtag_threshold=8,  # neither playlist is over-tagged
    )
    recs = service.recommend_playlists(
        TrackInput(
            track_id="t_blues",
            track_name="Gunpowder",
            artist="Sarah Dunn Music",
            genres=["Blues"],
        ),
        n=2,
    )
    by_id = {r.playlist_id: r.acceptance_probability for r in recs}
    # Real Blues playlist (primary overlap) must outrank the Rock playlist
    # whose only blues signal is via the subgenre.
    assert recs[0].playlist_id == "p_blues_primary"
    # And the gap should be at least the subgenre-only penalty (0.4x).
    assert by_id["p_rock_with_bluesrock"] <= by_id["p_blues_primary"] * 0.5


def test_explicit_genre_filter_overtagged_playlist_is_demoted(tmp_path: Path) -> None:
    """Fix G: a curator who selects every primary genre in the dropdown
    creates a 'genre soup' playlist that previously matched every track.
    The breadth penalty must demote it below a focused primary match."""
    pipe = _make_dummy_pipeline(PAIRWISE_FEATURE_COLS)
    save_bundle({"model": pipe, "feature_columns": PAIRWISE_FEATURE_COLS}, tmp_path)
    historical = pd.DataFrame(columns=["track_id", "playlist_id", "label", "track_name", "artist"])
    # 12 distinct primary tags -- way over a threshold of 4. Includes Blues.
    overtagged_genres = [
        "Pop", "Hip-Hop", "R&B", "Rock", "Electronic", "Country",
        "Jazz", "Classical", "Latin", "World", "Reggae", "Blues",
    ]
    playlists = pd.DataFrame(
        [
            {
                "playlist_id": "p_focused_blues",
                "playlist_name": "Real Blues",
                "description": "delta + chicago",
                "genres": ["Blues"],
                "subgenres": ["Delta Blues"],
            },
            {
                "playlist_id": "p_genre_soup",
                "playlist_name": "Daily Mix",
                "description": "any track goes",
                "genres": overtagged_genres,
                "subgenres": ["Lo-fi"],
            },
        ]
    )
    tracks = pd.DataFrame(columns=["track_id", "track_name", "artist", "bpm"])

    service = MatcherService(
        artifact_dir=str(tmp_path),
        historical_df=historical,
        playlists_df=playlists,
        tracks_df=tracks,
        text_embedder=_StubEmbedder(),
        hard_genre_filter=True,
        explicit_genre_no_match_penalty=0.01,
        explicit_genre_untagged_penalty=0.3,
        explicit_genre_subgenre_only_penalty=0.4,
        explicit_genre_broadtag_threshold=4,
    )
    recs = service.recommend_playlists(
        TrackInput(
            track_id="t_blues",
            track_name="Gunpowder",
            artist="Sarah Dunn Music",
            genres=["Blues"],
        ),
        n=2,
    )
    by_id = {r.playlist_id: r.acceptance_probability for r in recs}
    # Focused blues playlist must outrank the over-tagged catch-all even
    # though both have a primary `blues` overlap.
    assert recs[0].playlist_id == "p_focused_blues"
    # 12 primary tags -> breadth multiplier = 4/12 ≈ 0.33; focused playlist
    # is unscaled. So the over-tagged score should be ~0.33x of focused.
    ratio = by_id["p_genre_soup"] / by_id["p_focused_blues"]
    assert ratio <= 0.5, f"expected over-tagged ratio <= 0.5, got {ratio}"


def test_soft_attribute_penalty_disabled_when_user_inputs_empty(tmp_path: Path) -> None:
    """If the user supplies no soft inputs, the penalty must NOT fire even
    when the playlists have curator-set moods/etc."""
    pipe = _make_dummy_pipeline(PAIRWISE_FEATURE_COLS)
    save_bundle({"model": pipe, "feature_columns": PAIRWISE_FEATURE_COLS}, tmp_path)

    historical = pd.DataFrame(columns=["track_id", "playlist_id", "label", "track_name", "artist"])
    playlists = pd.DataFrame(
        [
            {
                "playlist_id": "p1",
                "playlist_name": "Hip-Hop A",
                "description": "trap",
                "moods": ["Energetic"],
            },
            {
                "playlist_id": "p2",
                "playlist_name": "Hip-Hop B",
                "description": "trap",
                "moods": ["Calm"],
            },
        ]
    )
    tracks = pd.DataFrame(columns=["track_id", "track_name", "artist", "bpm"])

    service = MatcherService(
        artifact_dir=str(tmp_path),
        historical_df=historical,
        playlists_df=playlists,
        tracks_df=tracks,
        text_embedder=_StubEmbedder(),
        hard_genre_filter=False,
        soft_attribute_penalty=0.4,
    )
    recs = service.recommend_playlists(
        TrackInput(track_id="t1", track_name="Trap Banger", artist="Newcomer"), n=2
    )
    # Both playlists kept their model probability untouched. They're tied
    # in this dummy model so it's enough to assert their probabilities are
    # equal (no penalty applied).
    assert recs[0].acceptance_probability == recs[1].acceptance_probability


def test_track_tier_filters_to_matching_playlists_only(tmp_path: Path) -> None:
    pipe = _make_dummy_pipeline(PAIRWISE_FEATURE_COLS)
    save_bundle({"model": pipe, "feature_columns": PAIRWISE_FEATURE_COLS}, tmp_path)

    historical = pd.DataFrame(columns=["track_id", "playlist_id", "label", "track_name", "artist"])
    playlists = pd.DataFrame(
        [
            {
                "playlist_id": "p_t1",
                "playlist_name": "Hip-Hop A",
                "description": "trap rap",
                "tier": 1,
            },
            {
                "playlist_id": "p_t2",
                "playlist_name": "Hip-Hop B",
                "description": "trap rap",
                "tier": 2,
            },
        ]
    )
    tracks = pd.DataFrame(columns=["track_id", "track_name", "artist", "bpm"])

    service = MatcherService(
        artifact_dir=str(tmp_path),
        historical_df=historical,
        playlists_df=playlists,
        tracks_df=tracks,
        text_embedder=_StubEmbedder(),
        hard_genre_filter=False,
    )
    recs = service.recommend_playlists(
        TrackInput(track_id="t1", track_name="Trap Banger", artist="Newcomer", tier=1),
        n=5,
    )
    assert len(recs) == 1
    assert recs[0].playlist_id == "p_t1"


def test_models_parse_track_tier() -> None:
    from matcher_agent.models import parse_track_tier

    assert parse_track_tier(None) is None
    assert parse_track_tier(3) == 3
    assert parse_track_tier("2") == 2

    with pytest.raises(ValueError):
        parse_track_tier(0)
    with pytest.raises(ValueError):
        parse_track_tier(5)
