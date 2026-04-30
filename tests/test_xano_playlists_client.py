from __future__ import annotations

from unittest.mock import Mock, patch

from matcher_agent.clients.xano_client import XanoClient


def test_playlists_are_flattened_and_paginated_with_next_page() -> None:
    page1_payload = {
        "items": [
            {
                "id": 292,
                "playlistName": "New Money Trap",
                "description": "desc",
                "playlistUrl": "https://open.spotify.com/playlist/2742U9cliLMSuxex35NtVC",
                "genres": ["Hip-Hop"],
                "subgenres": ["Trap", "Drill", "Trap"],
                "activity": ["workout", "party"],
                "countries": ["USA"],
                "languages": ["english", "english"],
                "tempos": ["fast"],
                "moods": ["energetic", "uplifting"],
            }
        ],
        "nextPage": 2,
    }
    page2_payload = {"items": [], "nextPage": None}

    response1 = Mock()
    response1.json.return_value = page1_payload
    response1.raise_for_status.return_value = None
    response2 = Mock()
    response2.json.return_value = page2_payload
    response2.raise_for_status.return_value = None

    with patch("matcher_agent.clients.xano_client.requests.get", side_effect=[response1, response2]) as mock_get:
        client = XanoClient(
            playlists_url="https://api.example.com/playlists",
            historical_matches_url="https://api.example.com/historical",
            page_size=100,
        )
        rows = client.fetch_playlists()

    assert len(rows) == 1
    assert rows[0]["playlist_id"] == "292"
    assert rows[0]["playlist_name"] == "New Money Trap"
    assert rows[0]["playlist_url"] == "https://open.spotify.com/playlist/2742U9cliLMSuxex35NtVC"
    # Genres pass through as a list; subgenres are deduplicated in order.
    assert rows[0]["genres"] == ["Hip-Hop"]
    assert rows[0]["subgenres"] == ["Trap", "Drill"]
    # Soft attributes pass through as deduplicated lists; raw values are
    # preserved (lowercasing/filtering of "any"/"other" happens later in
    # the profile builder so that it's safe to change those rules without
    # re-syncing from Xano).
    assert rows[0]["activity"] == ["workout", "party"]
    assert rows[0]["countries"] == ["USA"]
    assert rows[0]["languages"] == ["english"]
    assert rows[0]["tempos"] == ["fast"]
    assert rows[0]["moods"] == ["energetic", "uplifting"]
    assert mock_get.call_args_list[0].kwargs["params"]["page"] == 1
    assert mock_get.call_args_list[1].kwargs["params"]["page"] == 2


def test_playlists_handle_missing_genre_arrays() -> None:
    page1_payload = {
        "items": [
            {
                "id": 1,
                "playlistName": "Old Playlist",
                "description": "desc",
                "playlistUrl": "https://open.spotify.com/playlist/abc",
            }
        ],
        "nextPage": None,
    }
    response1 = Mock()
    response1.json.return_value = page1_payload
    response1.raise_for_status.return_value = None

    with patch("matcher_agent.clients.xano_client.requests.get", side_effect=[response1]):
        client = XanoClient(
            playlists_url="https://api.example.com/playlists",
            historical_matches_url="https://api.example.com/historical",
            page_size=100,
        )
        rows = client.fetch_playlists()

    assert rows[0]["genres"] == []
    assert rows[0]["subgenres"] == []
    # Missing soft-attribute arrays should also default to empty lists.
    assert rows[0]["activity"] == []
    assert rows[0]["countries"] == []
    assert rows[0]["languages"] == []
    assert rows[0]["tempos"] == []
    assert rows[0]["moods"] == []
