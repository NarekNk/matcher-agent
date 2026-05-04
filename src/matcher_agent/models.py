from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# Playlist / campaign tiers from Xano (1 = most selective; 4 = broadest).
TRACK_TIER_MIN = 1
TRACK_TIER_MAX = 4


def parse_track_tier(value: int | str | None) -> int | None:
    """Parse a request-time track tier: ``None`` if missing; 1–4 otherwise.

    Raises ``ValueError`` if a non-empty value is not a valid tier.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = int(text, 10)
        except ValueError as exc:
            raise ValueError(
                f"track tier must be an integer {TRACK_TIER_MIN}-{TRACK_TIER_MAX}, got {value!r}"
            ) from exc
    t = int(value)
    if t < TRACK_TIER_MIN or t > TRACK_TIER_MAX:
        raise ValueError(
            f"track tier must be in {TRACK_TIER_MIN}-{TRACK_TIER_MAX}, got {t}"
        )
    return t


def coerce_playlist_tier(value: Any) -> int | None:
    """Normalize a playlist ``tier`` cell from Xano/parquet to 1–4, or ``None``."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in ("nan", "none"):
            return None
        try:
            t = int(float(text))
        except ValueError:
            return None
    else:
        try:
            t = int(value)
        except (TypeError, ValueError):
            return None
    if TRACK_TIER_MIN <= t <= TRACK_TIER_MAX:
        return t
    return None


@dataclass
class TrackInput:
    track_id: str | None = None
    track_name: str = ""
    artist: str = ""
    album: str | None = None
    duration_ms: int | None = None
    preview_url: str | None = None
    spotify_url: str | None = None
    artist_genres: list[str] = field(default_factory=list)
    # Spotify popularity 0-100. None means "unknown" — popularity-fit
    # features will fall back to neutral values for this track.
    popularity: int | None = None
    # ----- Optional curator-style attributes supplied at prediction time -----
    # All are *optional*: leave the lists empty to opt out of the related
    # signals. These don't exist in the historical training data, so they
    # are NOT model features. The inference service uses them as a small
    # post-rerank multiplier (drops a candidate's score when the playlist's
    # curator-set attribute disagrees with the supplied track value).
    #
    # `genres` / `subgenres` follow the Xano vocabulary (top-level genre and
    # subgenre strings respectively). They feed the genre tag set used by
    # the hard genre filter and are mapped through `normalize_xano_labels`.
    genres: list[str] = field(default_factory=list)
    subgenres: list[str] = field(default_factory=list)
    activities: list[str] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    tempos: list[str] = field(default_factory=list)
    moods: list[str] = field(default_factory=list)
    # When set (1–4), only playlists with the same tier are candidates.
    # Not a model feature; enforced in ``MatcherService.recommend_playlists``.
    tier: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlaylistRecommendation:
    playlist_id: str
    playlist_name: str
    acceptance_probability: float
    rank: int


@dataclass
class MatchAttempt:
    match_id: str
    track_id: str
    playlist_id: str
    status: str
    updated_at: str | None = None
