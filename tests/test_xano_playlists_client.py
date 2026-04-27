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
    assert mock_get.call_args_list[0].kwargs["params"]["page"] == 1
    assert mock_get.call_args_list[1].kwargs["params"]["page"] == 2
