"""Qualitative sanity check for the matcher.

Loads the trained model + cached embeddings and scores synthetic
"new tracks" of known genres against the full catalog. Reports the
top-5 predicted playlists per track so we can eyeball whether the
genre matches.

Run:
    PYTHONPATH=src python scripts/qualitative_demo.py
"""
from __future__ import annotations

import json

from matcher_agent.config import get_settings
from matcher_agent.data.repository import DataRepository
from matcher_agent.embeddings import TextEmbedder
from matcher_agent.inference.service import MatcherService
from matcher_agent.models import TrackInput
from matcher_agent.storage.parquet_store import ParquetStore

# For the offline demo we manually inject artist_genres (these are what
# Spotify's `sp.artist()` would return at inference time).
DEMO_TRACKS: list[dict] = [
    {
        "label": "Pop track",
        "track": TrackInput(
            track_id="demo_pop_1",
            track_name="Blinding Lights",
            artist="The Weeknd",
            artist_genres=["pop", "dance pop", "synth pop"],
        ),
    },
    {
        "label": "Hip-hop / trap track",
        "track": TrackInput(
            track_id="demo_hh_1",
            track_name="HUMBLE.",
            artist="Kendrick Lamar",
            artist_genres=["hip hop", "rap", "west coast rap"],
        ),
    },
    {
        "label": "Country track",
        "track": TrackInput(
            track_id="demo_cn_1",
            track_name="Wagon Wheel",
            artist="Darius Rucker",
            artist_genres=["country", "country rock", "modern country"],
        ),
    },
    {
        "label": "EDM / house track",
        "track": TrackInput(
            track_id="demo_edm_1",
            track_name="Animals",
            artist="Martin Garrix",
            artist_genres=["big room", "edm", "electro house"],
        ),
    },
    {
        "label": "Latin reggaeton track",
        "track": TrackInput(
            track_id="demo_lat_1",
            track_name="Despacito",
            artist="Luis Fonsi, Daddy Yankee",
            artist_genres=["latin pop", "reggaeton", "latin"],
        ),
    },
    {
        "label": "Indie folk / acoustic track",
        "track": TrackInput(
            track_id="demo_folk_1",
            track_name="Skinny Love",
            artist="Bon Iver",
            artist_genres=["indie folk", "folk", "alternative folk"],
        ),
    },
]


def main() -> None:
    settings = get_settings()
    repo = DataRepository(ParquetStore(settings.data_dir))
    historical_df, _ = repo.load_labeled_historical_matches()
    playlists_df = repo.load_playlists()
    tracks_df = repo.load_tracks_from_export("output/training_data.csv")

    embedder = TextEmbedder(
        cache_path=settings.embeddings_dir / "text_embeddings.parquet",
        model_name=settings.text_embedding_model,
        device=settings.text_embedding_device,
    )
    service = MatcherService(
        artifact_dir=str(settings.model_dir),
        historical_df=historical_df,
        playlists_df=playlists_df,
        tracks_df=tracks_df,
        text_embedder=embedder,
        semantic_blend=settings.semantic_blend,
        hard_genre_filter=True,
    )

    print("\n" + "=" * 80)
    print("QUALITATIVE DEMO — top-5 playlists per genre archetype track")
    print("=" * 80)
    for case in DEMO_TRACKS:
        recs = service.recommend_playlists(case["track"], n=5)
        print(f"\n>>> {case['label']}: '{case['track'].track_name}' by {case['track'].artist}")
        for r in recs:
            print(
                f"    rank={r.rank}  p={r.acceptance_probability:.3f}  "
                f"id={r.playlist_id}  name={r.playlist_name!r}"
            )


if __name__ == "__main__":
    main()
