from __future__ import annotations

import re
from typing import Optional

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


def get_spotify_client(client_id: str, client_secret: str) -> spotipy.Spotify:
    auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return spotipy.Spotify(auth_manager=auth)


def fetch_playlist_tracks(
    sp: spotipy.Spotify,
    playlist_id: str,
    max_tracks: Optional[int] = None,
) -> list[dict]:
    tracks: list[dict] = []
    offset = 0
    while True:
        resp = sp.playlist_tracks(
            playlist_id,
            fields="items(track(id,name,artists,album,preview_url,duration_ms)),next",
            offset=offset,
            limit=50,
        )
        for item in resp["items"]:
            track = item.get("track")
            if not track or not track.get("id"):
                continue
            tracks.append(
                {
                    "track_id": track["id"],
                    "track_name": track["name"],
                    "artist": ", ".join(a["name"] for a in track["artists"]),
                    "album": track["album"]["name"],
                    "duration_ms": track["duration_ms"],
                    "preview_url": track.get("preview_url"),
                }
            )
            if max_tracks and len(tracks) >= max_tracks:
                return tracks
        if not resp["next"]:
            break
        offset += 50
    return tracks


def fetch_track_by_id(sp: spotipy.Spotify, track_id: str) -> dict:
    t = sp.track(track_id)
    artist_ids = [a.get("id") for a in t.get("artists", []) if a.get("id")]
    artist_genres: list[str] = []
    if artist_ids:
        try:
            for batch_start in range(0, len(artist_ids), 50):
                batch = artist_ids[batch_start : batch_start + 50]
                resp = sp.artists(batch)
                for art in resp.get("artists", []) or []:
                    artist_genres.extend(art.get("genres") or [])
        except Exception as exc:  # pragma: no cover - network-dependent
            print(f"[Spotify] Could not fetch artist genres: {exc}")
    return {
        "track_id": t["id"],
        "track_name": t["name"],
        "artist": ", ".join(a["name"] for a in t.get("artists", [])),
        "album": (t.get("album") or {}).get("name"),
        "duration_ms": t.get("duration_ms"),
        "preview_url": t.get("preview_url"),
        "spotify_url": (t.get("external_urls") or {}).get("spotify"),
        "artist_genres": sorted({g.lower() for g in artist_genres}),
    }


def parse_playlist_id(playlist_url_or_id: str) -> str:
    value = (playlist_url_or_id or "").strip()
    if not value:
        raise ValueError("playlist URL or id is empty")
    if "spotify.com/playlist/" not in value:
        return value
    match = re.search(r"playlist/([a-zA-Z0-9]+)", value)
    if not match:
        raise ValueError(f"cannot parse playlist id from '{playlist_url_or_id}'")
    return match.group(1)
