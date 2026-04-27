from __future__ import annotations

from unittest.mock import Mock, patch

from matcher_agent.clients.xano_client import XanoClient


def test_historical_matches_are_flattened_and_paginated_with_next_page() -> None:
    page1_payload = {
        "items": [
            {
                "id": 1,
                "campaigns_id": 100,
                "status": "closed",
                "playlists_id": [
                    {"id": 10, "playlistName": "P1", "playlistUrl": "https://open.spotify.com/playlist/aaa"}
                ],
                "track_info": {
                    "id": 100,
                    "trackName": "Song 1",
                    "artistName": "Artist 1",
                    "trackUrl": "https://open.spotify.com/track/t1",
                },
            }
        ],
        "nextPage": 2,
    }
    page2_payload = {
        "items": [
            {
                "id": 2,
                "campaigns_id": 101,
                "status": "declined",
                "playlists_id": [
                    {"id": 11, "playlistName": "P2", "playlistUrl": "https://open.spotify.com/playlist/bbb"}
                ],
                "track_info": {
                    "id": 101,
                    "trackName": "Song 2",
                    "artistName": "Artist 2",
                    "trackUrl": "https://open.spotify.com/track/t2",
                },
            }
        ],
        "nextPage": None,
    }

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
            page_size=25,
        )
        rows = client.fetch_historical_matches()

    assert len(rows) == 2
    assert rows[0]["match_id"] == "1"
    assert rows[0]["playlist_id"] == "10"
    # Canonical track_id is now the spotify_track_id parsed from track_url.
    assert rows[0]["spotify_track_id"] == "t1"
    assert rows[0]["track_id"] == "t1"
    assert rows[0]["xano_track_id"] == "100"
    assert rows[0]["status"] == "closed"
    assert rows[1]["match_id"] == "2"
    assert rows[1]["playlist_id"] == "11"
    assert rows[1]["track_id"] == "t2"

    first_call_params = mock_get.call_args_list[0].kwargs["params"]
    second_call_params = mock_get.call_args_list[1].kwargs["params"]
    assert first_call_params["page"] == 1
    assert second_call_params["page"] == 2
