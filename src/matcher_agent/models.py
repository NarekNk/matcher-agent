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
