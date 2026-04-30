from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
