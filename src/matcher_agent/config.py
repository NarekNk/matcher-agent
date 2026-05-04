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
    semantic_blend: float = float(os.getenv("SEMANTIC_BLEND", "0.25"))
    hard_genre_filter: bool = os.getenv("HARD_GENRE_FILTER", "1") not in {"0", "false", "False"}
    # Random/hard-negative pairs sampled per accepted positive during training.
    # Must be > 0 to teach the model what an "obvious mismatch" looks like —
    # otherwise it only learns to discriminate among historical pitches, which
    # are already genre-controlled by curators.
    negative_sample_ratio: float = float(os.getenv("NEGATIVE_SAMPLE_RATIO", "5.0"))
    # Fraction of the negative budget that must come from genre-conflicting
    # playlists (zero canonical-tag overlap with the track). The remainder is
    # uniform-random catalog negatives. 0 disables hard-negative mining.
    negative_conflict_fraction: float = float(os.getenv("NEGATIVE_CONFLICT_FRACTION", "0.5"))
    # Multiplicative penalty applied at inference time to a playlist's
    # acceptance probability for each soft attribute (mood, language,
    # activity, country, tempo) where the user-supplied track value and the
    # playlist's curator-set value disagree (both non-empty, no overlap).
    # Range (0, 1]; 1.0 disables the penalty entirely. Default 0.7 means
    # ~30% drop per conflicting attribute, capping at ~0.17 with all 5 in
    # disagreement -- a noticeable but not fatal reweighting.
    soft_attribute_penalty: float = float(os.getenv("SOFT_ATTRIBUTE_PENALTY", "0.7"))
    # Stricter multiplier used only for `languages` mismatches (e.g. English
    # track vs Portuguese-tagged playlist). Same semantics as
    # `soft_attribute_penalty`: applied once per language conflict when both
    # sides have non-empty normalized language sets and they do not overlap.
    language_mismatch_penalty: float = float(
        os.getenv("LANGUAGE_MISMATCH_PENALTY", "0.3")
    )
    # When the user explicitly supplies track genres/subgenres, we switch
    # to a strict positive-overlap filter:
    #   * playlists whose Xano tags do not share any tag with the supplied
    #     ones are multiplied by EXPLICIT_GENRE_NO_MATCH_PENALTY
    #     (default 0.02 -- effectively dropped from top-N).
    #   * playlists that have no Xano tags at all (we can't verify fit)
    #     are multiplied by EXPLICIT_GENRE_UNTAGGED_PENALTY
    #     (default 0.3 -- down-weighted but reachable).
    explicit_genre_no_match_penalty: float = float(
        os.getenv("EXPLICIT_GENRE_NO_MATCH_PENALTY", "0.02")
    )
    explicit_genre_untagged_penalty: float = float(
        os.getenv("EXPLICIT_GENRE_UNTAGGED_PENALTY", "0.3")
    )
    # Tier penalty applied when the user's explicit track tags overlap a
    # playlist's secondary (subgenre/text-derived) tags but do NOT overlap
    # any of its primary Xano `genres` array entries. Example: a Rock
    # playlist with subgenre "Blues Rock" would match a Blues track only
    # via the subgenre, so it ranks below a real Blues primary playlist.
    # Range (0, 1]; 1.0 disables the tier (any overlap counts as a primary
    # match again).
    explicit_genre_subgenre_only_penalty: float = float(
        os.getenv("EXPLICIT_GENRE_SUBGENRE_ONLY_PENALTY", "0.4")
    )
    # Over-tagging "broadtag" penalty: when the user supplies explicit
    # genres, playlists whose primary Xano `genres` array exceeds this
    # threshold are treated as generic catch-all mixes and scaled by
    # `threshold / len(primary_tags)`. This prevents curators who select
    # every available genre on the dropdown from passing the explicit
    # filter on every track. Set to a large number (e.g. 999) to disable.
    explicit_genre_broadtag_threshold: int = int(
        os.getenv("EXPLICIT_GENRE_BROADTAG_THRESHOLD", "4")
    )
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
