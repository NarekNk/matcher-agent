from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import requests

from matcher_agent.models import coerce_playlist_tier


@dataclass
class XanoClient:
    playlists_url: str
    historical_matches_url: str
    timeout_s: float = 20.0
    page_size: int = 200
    historical_max_pages: int | None = None

    @staticmethod
    def _spotify_track_id_from_url(track_url: str | None) -> str | None:
        if not track_url:
            return None
        match = re.search(r"spotify\.com/track/([a-zA-Z0-9]+)", track_url)
        if not match:
            return None
        return match.group(1)

    @staticmethod
    def _as_str_or_none(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text if text else None

    @staticmethod
    def _as_str_list(value: Any) -> list[str]:
        """Normalize a string-array column (e.g. Xano `genres`/`subgenres`).

        Returns a deduplicated list of stripped strings, preserving order.
        Anything that isn't a list, or whose elements can't be stringified
        into a non-empty value, is dropped.
        """
        if not isinstance(value, list):
            return []
        seen: set[str] = set()
        out: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out

    def _fetch_endpoint(
        self,
        base_url: str,
        *,
        endpoint_name: str,
        updated_after: str | None = None,
        max_pages: int | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        page = 1
        pages_fetched = 0
        print(f"[Xano:{endpoint_name}] starting fetch (page_size={self.page_size})")
        while True:
            # Explicitly include page in query so Xano returns paginated slices.
            params = {"page": page, "per_page": self.page_size}
            if updated_after:
                params["updated_after"] = updated_after
            resp = requests.get(base_url, params=params, timeout=self.timeout_s)
            resp.raise_for_status()
            payload = resp.json()

            if isinstance(payload, dict):
                items = payload.get("items") or payload.get("data") or payload.get("results") or []
                next_page = payload.get("nextPage")
            elif isinstance(payload, list):
                items = payload
                next_page = None
            else:
                items = []
                next_page = None

            if not items:
                print(f"[Xano:{endpoint_name}] page={page} returned 0 items; stopping.")
                break

            rows.extend(items)
            pages_fetched += 1
            print(f"[Xano:{endpoint_name}] page={page} fetched={len(items)} total={len(rows)}")

            if max_pages is not None and max_pages > 0 and pages_fetched >= max_pages:
                print(f"[Xano:{endpoint_name}] reached max_pages={max_pages}; stopping early.")
                break

            # Prefer Xano's pagination cursor when present.
            if isinstance(next_page, int):
                page = next_page
                continue

            if len(items) < self.page_size:
                break
            page += 1
        print(f"[Xano:{endpoint_name}] completed total_rows={len(rows)}")
        return rows

    def fetch_playlists(self, *, updated_after: str | None = None) -> list[dict[str, Any]]:
        rows = self._fetch_endpoint(self.playlists_url, endpoint_name="playlists", updated_after=updated_after)
        normalized: list[dict[str, Any]] = []
        for row in rows:
            normalized.append(
                {
                    "playlist_id": self._as_str_or_none(row.get("id")),
                    "playlist_name": self._as_str_or_none(row.get("playlistName")),
                    "description": self._as_str_or_none(row.get("description")),
                    "playlist_url": self._as_str_or_none(row.get("playlistUrl")),
                    # Curator-tagged genre arrays. Stored as list[str]; downstream
                    # uses `genre_normalizer.normalize_xano_labels()` to map to
                    # canonical tags.
                    "genres": self._as_str_list(row.get("genres")),
                    "subgenres": self._as_str_list(row.get("subgenres")),
                    # Soft attributes. Stored as raw list[str] -- the
                    # downstream `attribute_normalizer.normalize_attribute_labels`
                    # drops "any"/"other"/null and lowercases the rest. We keep
                    # the raw values here so older normalization rules can
                    # change without forcing a full re-sync from Xano.
                    "activity": self._as_str_list(row.get("activity")),
                    "countries": self._as_str_list(row.get("countries")),
                    "languages": self._as_str_list(row.get("languages")),
                    "tempos": self._as_str_list(row.get("tempos")),
                    "moods": self._as_str_list(row.get("moods")),
                    # 1–4 from Xano; missing/invalid -> None (stored as null in parquet).
                    "tier": coerce_playlist_tier(row.get("tier")),
                }
            )
        print(f"[Xano:playlists] normalized_rows={len(normalized)}")
        return normalized

    def fetch_historical_matches(self, *, updated_after: str | None = None) -> list[dict[str, Any]]:
        rows = self._fetch_endpoint(
            self.historical_matches_url,
            endpoint_name="historical_matches",
            updated_after=updated_after,
            max_pages=self.historical_max_pages,
        )
        normalized: list[dict[str, Any]] = []

        def _flatten_playlist_items(raw: Any) -> list[dict[str, Any]]:
            """Flatten mixed nested playlist payloads into playlist dicts."""
            if raw is None:
                return []
            if isinstance(raw, dict):
                return [raw]
            if isinstance(raw, list):
                out: list[dict[str, Any]] = []
                for item in raw:
                    out.extend(_flatten_playlist_items(item))
                return out
            return []

        for row in rows:
            track_info = row.get("track_info") or {}
            track_url = self._as_str_or_none(track_info.get("trackUrl"))
            spotify_track_id = self._spotify_track_id_from_url(track_url)
            xano_track_id = self._as_str_or_none(track_info.get("id") or row.get("campaigns_id"))
            canonical_track_id = self._as_str_or_none(spotify_track_id or xano_track_id)
            playlists = _flatten_playlist_items(row.get("playlists_id"))
            if not playlists:
                normalized.append(
                    {
                        "match_id": self._as_str_or_none(row.get("id")),
                        "campaign_id": self._as_str_or_none(row.get("campaigns_id")),
                        "playlist_id": None,
                        "playlist_name": None,
                        "playlist_url": None,
                        "status": self._as_str_or_none(row.get("status")),
                        "track_id": canonical_track_id,
                        "spotify_track_id": spotify_track_id,
                        "xano_track_id": xano_track_id,
                        "track_name": self._as_str_or_none(track_info.get("trackName")),
                        "artist": self._as_str_or_none(track_info.get("artistName")),
                        "track_url": track_url,
                    }
                )
                continue

            for playlist in playlists:
                normalized.append(
                    {
                        "match_id": self._as_str_or_none(row.get("id")),
                        "campaign_id": self._as_str_or_none(row.get("campaigns_id")),
                        "playlist_id": self._as_str_or_none(playlist.get("id")),
                        "playlist_name": self._as_str_or_none(playlist.get("playlistName")),
                        "playlist_url": self._as_str_or_none(playlist.get("playlistUrl")),
                        "status": self._as_str_or_none(row.get("status")),
                        "track_id": canonical_track_id,
                        "spotify_track_id": spotify_track_id,
                        "xano_track_id": xano_track_id,
                        "track_name": self._as_str_or_none(track_info.get("trackName")),
                        "artist": self._as_str_or_none(track_info.get("artistName")),
                        "track_url": track_url,
                    }
                )
        print(f"[Xano:historical_matches] normalized_rows={len(normalized)}")
        return normalized
