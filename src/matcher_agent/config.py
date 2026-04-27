from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    spotify_client_id: str | None = os.getenv("SPOTIFY_CLIENT_ID")
    spotify_client_secret: str | None = os.getenv("SPOTIFY_CLIENT_SECRET")
    preview_resolver_url: str | None = os.getenv("PREVIEW_RESOLVER_URL")
    xano_playlists_url: str | None = os.getenv("XANO_PLAYLISTS_URL")
    xano_historical_matches_url: str | None = os.getenv("XANO_HISTORICAL_MATCHES_URL")
    xano_page_size: int = int(os.getenv("XANO_PAGE_SIZE", "200"))
    xano_timeout_s: float = float(os.getenv("XANO_TIMEOUT_S", "20"))
    xano_historical_max_pages: int = int(os.getenv("XANO_HISTORICAL_MAX_PAGES", "0"))
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "output"))
    audio_dir: Path = Path(os.getenv("AUDIO_DIR", "audio_previews"))
    model_dir: Path = Path(os.getenv("MODEL_DIR", "artifacts"))
    embeddings_dir: Path = Path(os.getenv("EMBEDDINGS_DIR", "data/embeddings"))
    text_embedding_model: str = os.getenv("TEXT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    text_embedding_device: str | None = os.getenv("TEXT_EMBEDDING_DEVICE") or None
    semantic_blend: float = float(os.getenv("SEMANTIC_BLEND", "0.5"))
    hard_genre_filter: bool = os.getenv("HARD_GENRE_FILTER", "1") not in {"0", "false", "False"}
    negative_sample_ratio: float = float(os.getenv("NEGATIVE_SAMPLE_RATIO", "0.0"))
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))
    max_playlists: int = int(os.getenv("MAX_PLAYLISTS", "500"))
    max_tracks_per_playlist: int = int(os.getenv("MAX_TRACKS_PER_PLAYLIST", "30"))
    feature_download_concurrency: int = int(os.getenv("FEATURE_DOWNLOAD_CONCURRENCY", "16"))
    feature_analysis_workers: int | None = (
        int(v) if (v := os.getenv("FEATURE_ANALYSIS_WORKERS")) else None
    )
    feature_progress_every: int = int(os.getenv("FEATURE_PROGRESS_EVERY", "25"))


def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.audio_dir.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    settings.embeddings_dir.mkdir(parents=True, exist_ok=True)
    return settings
