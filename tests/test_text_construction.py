from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from matcher_agent.embeddings.text_embedder import (
    TextEmbedder,
    _TEXT_FORMAT_VERSION,
    _stable_text_hash,
)
from matcher_agent.features.playlist_profiles import (
    build_playlist_text,
    build_playlist_text_strings,
    build_track_text,
    build_track_text_strings,
)


# ---------------------------------------------------------------------------
# build_track_text
# ---------------------------------------------------------------------------


def test_track_text_base_only() -> None:
    text = build_track_text(artist="MC Foo", track_name="Drill Anthem")
    assert text == "MC Foo - Drill Anthem"


def test_track_text_with_genres() -> None:
    text = build_track_text(
        artist="MC Foo",
        track_name="Drill Anthem",
        artist_genres=["hip hop", "trap"],
        genres=["Hip-Hop"],
    )
    assert "Genres:" in text
    assert "hip hop" in text
    assert "trap" in text
    assert "hip-hop" in text


def test_track_text_deduplicates_genres() -> None:
    text = build_track_text(
        artist="MC Foo",
        track_name="Drill Anthem",
        artist_genres=["hip hop"],
        genres=["Hip Hop"],
    )
    assert text.lower().count("hip hop") == 1


def test_track_text_with_language_and_mood() -> None:
    text = build_track_text(
        artist="A",
        track_name="B",
        languages=["english"],
        moods=["energetic"],
    )
    assert "Language: english." in text
    assert "Mood: energetic." in text


def test_track_text_missing_columns_graceful() -> None:
    text = build_track_text(artist="Solo Artist", track_name="")
    assert text == "Solo Artist"


def test_track_text_from_dataframe_row() -> None:
    df = pd.DataFrame(
        [
            {
                "track_id": "t1",
                "artist": "MC Foo",
                "track_name": "Drill",
                "artist_genres": ["hip hop", "trap"],
            }
        ]
    )
    strings = build_track_text_strings(df)
    assert len(strings) == 1
    assert "Genres:" in strings[0]
    assert "hip hop" in strings[0]


def test_track_text_from_dataframe_without_genre_columns() -> None:
    df = pd.DataFrame(
        [{"track_id": "t1", "artist": "MC Foo", "track_name": "Drill"}]
    )
    strings = build_track_text_strings(df)
    assert strings == ["MC Foo - Drill"]


# ---------------------------------------------------------------------------
# build_playlist_text
# ---------------------------------------------------------------------------


def test_playlist_text_base_only() -> None:
    text = build_playlist_text(
        playlist_name="Hip-Hop Hits", description="rap and trap"
    )
    assert text == "Hip-Hop Hits. rap and trap"


def test_playlist_text_with_genres_and_moods() -> None:
    text = build_playlist_text(
        playlist_name="Workout Mix",
        description="high energy",
        genres=["Pop", "Hip-Hop"],
        moods=["Energetic"],
        activities=["Workout"],
    )
    assert "Genres: pop, hip-hop." in text
    assert "Mood: energetic." in text
    assert "Activity: workout." in text


def test_playlist_text_no_description() -> None:
    text = build_playlist_text(playlist_name="Daily Mix")
    assert text == "Daily Mix"


def test_playlist_text_from_dataframe_with_xano_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "playlist_id": "p1",
                "playlist_name": "Chill Vibes",
                "description": "relax",
                "genres": ["Jazz", "Blues"],
                "moods": ["Calm"],
                "activity": ["Relax"],
            }
        ]
    )
    strings = build_playlist_text_strings(df)
    assert len(strings) == 1
    assert "Genres: jazz, blues." in strings[0]
    assert "Mood: calm." in strings[0]
    assert "Activity: relax." in strings[0]


# ---------------------------------------------------------------------------
# Train/serve text parity
# ---------------------------------------------------------------------------


def test_track_text_parity_with_inference() -> None:
    """The same metadata must produce identical text whether it comes from
    a DataFrame row (training path) or from TrackInput fields (inference
    path via build_track_text kwargs)."""
    row_text = build_track_text(
        artist="Sarah Dunn Music",
        track_name="Gunpowder",
        genres=["Blues"],
        subgenres=["Soul", "Jazz"],
        languages=["english"],
    )
    assert "Genres: blues, soul, jazz." in row_text
    assert "Language: english." in row_text


# ---------------------------------------------------------------------------
# Cache: text format version in hash
# ---------------------------------------------------------------------------


def test_text_hash_includes_format_version() -> None:
    h1 = _stable_text_hash("some text")
    assert len(h1) == 40  # SHA1 hex length
    assert _stable_text_hash("some text") == h1  # deterministic


def test_text_hash_changes_with_version(monkeypatch) -> None:
    import matcher_agent.embeddings.text_embedder as mod

    h_v2 = _stable_text_hash("hello world")
    original = mod._TEXT_FORMAT_VERSION
    monkeypatch.setattr(mod, "_TEXT_FORMAT_VERSION", "v99")
    h_v99 = mod._stable_text_hash("hello world")
    monkeypatch.setattr(mod, "_TEXT_FORMAT_VERSION", original)
    assert h_v2 != h_v99


# ---------------------------------------------------------------------------
# Cache: model-name keyed file path
# ---------------------------------------------------------------------------


def test_resolved_cache_path_includes_model_name(tmp_path: Path) -> None:
    embedder = TextEmbedder(
        tmp_path / "text_embeddings.parquet",
        model_name="all-MiniLM-L6-v2",
    )
    resolved = embedder._resolved_cache_path
    assert "all-MiniLM-L6-v2" in resolved.name
    assert resolved.suffix == ".parquet"


def test_resolved_cache_path_handles_slash_in_model_name(tmp_path: Path) -> None:
    embedder = TextEmbedder(
        tmp_path / "text_embeddings.parquet",
        model_name="BAAI/bge-small-en-v1.5",
    )
    resolved = embedder._resolved_cache_path
    assert "/" not in resolved.name
    assert "BAAI--bge-small-en-v1.5" in resolved.name


def test_different_models_get_different_cache_files(tmp_path: Path) -> None:
    base = tmp_path / "emb.parquet"
    e1 = TextEmbedder(base, model_name="all-MiniLM-L6-v2")
    e2 = TextEmbedder(base, model_name="all-mpnet-base-v2")
    assert e1._resolved_cache_path != e2._resolved_cache_path


# ---------------------------------------------------------------------------
# dim property for known models
# ---------------------------------------------------------------------------


def test_dim_default_models() -> None:
    e1 = TextEmbedder(Path("x.parquet"), model_name="all-MiniLM-L6-v2")
    assert e1.dim == 384
    e2 = TextEmbedder(Path("x.parquet"), model_name="all-mpnet-base-v2")
    assert e2.dim == 768
    e3 = TextEmbedder(Path("x.parquet"), model_name="BAAI/bge-small-en-v1.5")
    assert e3.dim == 384
