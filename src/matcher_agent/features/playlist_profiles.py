from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from matcher_agent.features.attribute_normalizer import (
    SOFT_ATTRIBUTE_NAMES,
    normalize_attribute_labels,
)
from matcher_agent.features.genre_normalizer import normalize_xano_labels
from matcher_agent.features.genre_tagger import tag_text
from matcher_agent.models import coerce_playlist_tier

# Audio columns used to build per-playlist accepted-track centroids.
# Order matters: new columns are appended at the end so that legacy
# cached data (which only has the first 21 columns) can be loaded and
# extended with NaN for the missing tail.
AUDIO_FEATURE_COLS: tuple[str, ...] = (
    # --- original 21 features (v1) ---
    "bpm",
    "loudness",
    "danceability",
    "energy",
    "spectral_centroid",
    "spectral_rolloff",
    "spectral_flux",
    "zcr",
    "mfcc_1",
    "mfcc_2",
    "mfcc_3",
    "mfcc_4",
    "mfcc_5",
    "mfcc_6",
    "mfcc_7",
    "mfcc_8",
    "mfcc_9",
    "mfcc_10",
    "mfcc_11",
    "mfcc_12",
    "mfcc_13",
    # --- v2: key, mode, onset rate ---
    "key",
    "mode",
    "onset_rate",
    # --- v2: temporal variation (frame-level std dev) ---
    "spectral_centroid_std",
    "spectral_rolloff_std",
    "spectral_flux_std",
    "zcr_std",
    "mfcc_1_std",
    "mfcc_2_std",
    "mfcc_3_std",
    "mfcc_4_std",
    "mfcc_5_std",
    "mfcc_6_std",
    "mfcc_7_std",
    "mfcc_8_std",
    "mfcc_9_std",
    "mfcc_10_std",
    "mfcc_11_std",
    "mfcc_12_std",
    "mfcc_13_std",
)


def ensure_audio_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill any missing AUDIO_FEATURE_COLS with NaN for backward compat.

    Old cached data (CSV / parquet) may not contain the v2 columns
    (key, mode, onset_rate, *_std). Adding them as NaN lets downstream
    code (SimpleImputer, profile builder) handle them gracefully.
    """
    missing = [c for c in AUDIO_FEATURE_COLS if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = np.nan
    return df


def _dedup_terms(*sources: list[str] | None) -> list[str]:
    """Merge multiple label lists into a deduplicated, lowercased list."""
    seen: set[str] = set()
    out: list[str] = []
    for source in sources:
        if not source:
            continue
        for term in source:
            t = (term or "").strip().lower()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    return out


def build_track_text(
    *,
    artist: str = "",
    track_name: str = "",
    artist_genres: list[str] | None = None,
    genres: list[str] | None = None,
    subgenres: list[str] | None = None,
    languages: list[str] | None = None,
    moods: list[str] | None = None,
) -> str:
    """Build the text string that gets embedded for a track.

    Shared between training (DataFrame rows) and inference (TrackInput).
    Genre/language/mood metadata is appended when available so the text
    embedding captures genre-awareness — matching the inference-time
    enrichment that was previously only done in ``_track_text_for_input``.
    """
    artist = (artist or "").strip()
    track_name = (track_name or "").strip()
    base = f"{artist} - {track_name}" if (artist and track_name) else (track_name or artist)

    suffix_parts: list[str] = []
    genre_terms = _dedup_terms(artist_genres, genres, subgenres)
    if genre_terms:
        suffix_parts.append(f"Genres: {', '.join(genre_terms)}.")
    lang_terms = _dedup_terms(languages)
    if lang_terms:
        suffix_parts.append(f"Language: {', '.join(lang_terms)}.")
    mood_terms = _dedup_terms(moods)
    if mood_terms:
        suffix_parts.append(f"Mood: {', '.join(mood_terms)}.")
    if suffix_parts:
        return f"{base}. " + " ".join(suffix_parts)
    return base


def build_playlist_text(
    *,
    playlist_name: str = "",
    description: str = "",
    genres: list[str] | None = None,
    subgenres: list[str] | None = None,
    moods: list[str] | None = None,
    activities: list[str] | None = None,
) -> str:
    """Build the text string that gets embedded for a playlist.

    Genre/mood/activity metadata from Xano enriches the description so
    the text embedding reflects the curator's intent, not just the
    (often generic) playlist name.
    """
    name = (playlist_name or "").strip()
    desc = (description or "").strip()
    if desc and desc.lower() != "nan":
        base = f"{name}. {desc}"
    else:
        base = name

    suffix_parts: list[str] = []
    genre_terms = _dedup_terms(genres, subgenres)
    if genre_terms:
        suffix_parts.append(f"Genres: {', '.join(genre_terms)}.")
    mood_terms = _dedup_terms(moods)
    if mood_terms:
        suffix_parts.append(f"Mood: {', '.join(mood_terms)}.")
    act_terms = _dedup_terms(activities)
    if act_terms:
        suffix_parts.append(f"Activity: {', '.join(act_terms)}.")
    if suffix_parts:
        return f"{base}. " + " ".join(suffix_parts)
    return base


def _track_text(row: pd.Series) -> str:
    return build_track_text(
        artist=str(row.get("artist") or ""),
        track_name=str(row.get("track_name") or ""),
        artist_genres=_coerce_label_list(row.get("artist_genres")),
        genres=_coerce_label_list(row.get("genres")),
        subgenres=_coerce_label_list(row.get("subgenres")),
        languages=_coerce_label_list(row.get("languages")),
        moods=_coerce_label_list(row.get("moods")),
    )


def _playlist_text(row: pd.Series) -> str:
    return build_playlist_text(
        playlist_name=str(row.get("playlist_name") or ""),
        description=str(row.get("description") or ""),
        genres=_coerce_label_list(row.get("genres")),
        subgenres=_coerce_label_list(row.get("subgenres")),
        moods=_coerce_label_list(row.get("moods")),
        activities=_coerce_label_list(row.get("activity")),
    )


def _safe_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        return vec
    return vec / norm


def _coerce_label_list(value) -> list[str]:
    """Best-effort coerce a parquet/csv cell into a clean list[str].

    Pandas may hand us actual list objects (parquet list-typed columns)
    or stringified lists when the value round-tripped through CSV. Both
    are handled. Empty / null values yield an empty list.
    """
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in ("nan", "none", "[]"):
            return []
        # Best-effort: looks like a JSON-encoded list.
        if text.startswith("[") and text.endswith("]"):
            try:
                import json

                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if str(v).strip()]
            except Exception:
                pass
        # Fallback: comma-separated.
        return [piece.strip() for piece in text.split(",") if piece.strip()]
    return []


@dataclass
class PlaylistProfile:
    playlist_id: str
    playlist_name: str
    text_emb: np.ndarray
    semantic_centroid: np.ndarray
    accepted_count: int
    declined_count: int
    acceptance_rate: float
    audio_centroid: np.ndarray | None
    audio_std: np.ndarray | None
    tags: set[str] = field(default_factory=set)
    # Canonical tags derived from the Xano top-level `genres` array only
    # (i.e. the curator's *primary* genre selection, NOT subgenres or text-
    # extracted tags). Used by the strict explicit-genre filter at inference
    # time to differentiate "this playlist is genuinely a Blues playlist"
    # from "this rock playlist happens to have 'Blues Rock' as a subgenre".
    primary_tags: set[str] = field(default_factory=set)
    # Per-dimension min/max across accepted tracks' audio vectors. Used
    # by the audio_range_ratio feature to check whether a candidate track
    # falls inside the playlist's observed audio envelope.
    audio_min: np.ndarray | None = None
    audio_max: np.ndarray | None = None
    # Track-popularity statistics across this playlist's accepted tracks.
    # `None` means we have no popularity data (no accepted tracks with
    # popularity recorded). Used by the popularity-fit features.
    popularity_mean: float | None = None
    popularity_std: float | None = None
    popularity_count: int = 0
    # Curator-supplied "soft" attributes from the Xano playlist payload.
    # An empty set means "no preference" (the curator selected "any"/"other"
    # or left it null). Used by the inference-time soft-attribute penalty,
    # never by the trained model (we have no track-side training data for
    # these attributes, so they cannot be learned).
    activities: set[str] = field(default_factory=set)
    countries: set[str] = field(default_factory=set)
    languages: set[str] = field(default_factory=set)
    tempos: set[str] = field(default_factory=set)
    moods: set[str] = field(default_factory=set)
    # Xano playlist tier 1–4; ``None`` if missing or invalid in source data.
    tier: int | None = None

    def soft_attribute_sets(self) -> dict[str, set[str]]:
        """Return the soft-attribute sets keyed by canonical name."""
        return {
            "activities": self.activities,
            "countries": self.countries,
            "languages": self.languages,
            "tempos": self.tempos,
            "moods": self.moods,
        }


@dataclass
class ProfileBundle:
    """Container for everything inference and training need about playlists."""

    profiles: dict[str, PlaylistProfile]
    audio_feature_cols: list[str]
    embedding_dim: int

    def as_frame(self) -> pd.DataFrame:
        rows = []
        for pid, prof in self.profiles.items():
            rows.append(
                {
                    "playlist_id": pid,
                    "playlist_name": prof.playlist_name,
                    "playlist_accepted_track_count": prof.accepted_count,
                    "playlist_declined_track_count": prof.declined_count,
                    "playlist_acceptance_rate": prof.acceptance_rate,
                    "playlist_tag_count": len(prof.tags),
                    "playlist_text_emb": prof.text_emb,
                    "playlist_semantic_centroid": prof.semantic_centroid,
                    "playlist_audio_centroid": prof.audio_centroid,
                    "playlist_audio_std": prof.audio_std,
                    "playlist_audio_min": prof.audio_min,
                    "playlist_audio_max": prof.audio_max,
                    "playlist_tags": prof.tags,
                    "playlist_popularity_mean": prof.popularity_mean,
                    "playlist_popularity_std": prof.popularity_std,
                    "playlist_popularity_count": prof.popularity_count,
                }
            )
        return pd.DataFrame(rows)


def build_playlist_text_strings(playlists_df: pd.DataFrame) -> list[str]:
    return [_playlist_text(row) for _, row in playlists_df.iterrows()]


def build_track_text_strings(tracks_df: pd.DataFrame) -> list[str]:
    return [_track_text(row) for _, row in tracks_df.iterrows()]


def build_track_popularity_lookup(tracks_df: pd.DataFrame) -> dict[str, float]:
    """Build a track_id -> popularity (0-100, float) lookup. Tracks without
    popularity (NaN) are simply absent from the dict."""
    if "popularity" not in tracks_df.columns:
        return {}
    sub = tracks_df[["track_id", "popularity"]].copy()
    sub["track_id"] = sub["track_id"].astype("string")
    sub["popularity"] = pd.to_numeric(sub["popularity"], errors="coerce")
    sub = sub.dropna(subset=["track_id", "popularity"])
    return {str(row["track_id"]): float(row["popularity"]) for _, row in sub.iterrows()}


def build_profiles(
    playlists_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    tracks_df: pd.DataFrame,
    *,
    track_text_emb_by_id: dict[str, np.ndarray],
    playlist_text_emb_by_id: dict[str, np.ndarray],
    label_col: str = "label",
    audio_feature_cols: tuple[str, ...] = AUDIO_FEATURE_COLS,
    semantic_blend: float = 0.25,
) -> ProfileBundle:
    """Compute one PlaylistProfile per playlist in `playlists_df`.

    Tag assembly: each playlist's `tags` is the union of:
      1. canonical tags from Xano `genres`/`subgenres` arrays (authoritative)
      2. canonical tags from a regex pass on `playlist_name + description`

    semantic_blend controls how much the playlist's own text embedding weighs
    relative to the centroid of its historically accepted tracks:
      profile = blend * playlist_text + (1 - blend) * accepted_track_centroid.
    When a playlist has no accepted tracks yet, profile = playlist_text.
    """
    print(f"[Profiles] Building profiles for {len(playlists_df)} playlists.")
    matches = matches_df.copy()
    matches["playlist_id"] = matches["playlist_id"].astype("string")
    matches["track_id"] = matches["track_id"].astype("string")

    accepted = matches[matches[label_col] == 1]
    declined = matches[matches[label_col] == 0]
    accepted_by_pl = accepted.groupby("playlist_id")["track_id"].apply(list).to_dict()
    declined_count_by_pl = declined.groupby("playlist_id").size().to_dict()
    accepted_count_by_pl = accepted.groupby("playlist_id").size().to_dict()

    audio_lookup: dict[str, np.ndarray] = {}
    available_audio_cols = [c for c in audio_feature_cols if c in tracks_df.columns]
    if available_audio_cols:
        tracks_a = tracks_df[["track_id", *available_audio_cols]].copy()
        tracks_a["track_id"] = tracks_a["track_id"].astype("string")
        for _, row in tracks_a.iterrows():
            tid = row["track_id"]
            if pd.isna(tid):
                continue
            vec = np.array(
                [pd.to_numeric(row[c], errors="coerce") for c in available_audio_cols],
                dtype=np.float64,
            )
            if not np.all(np.isnan(vec)):
                audio_lookup[str(tid)] = vec

    popularity_lookup = build_track_popularity_lookup(tracks_df)

    profiles: dict[str, PlaylistProfile] = {}
    embedding_dim = next(iter(playlist_text_emb_by_id.values())).shape[0] if playlist_text_emb_by_id else 384

    has_xano_genres = "genres" in playlists_df.columns
    has_xano_subgenres = "subgenres" in playlists_df.columns

    # Soft-attribute columns. Each may be missing on older playlist parquet
    # files; we default to an empty set in that case.
    soft_attr_columns: dict[str, str] = {
        "activities": "activity",
        "countries": "countries",
        "languages": "languages",
        "tempos": "tempos",
        "moods": "moods",
    }
    soft_attr_present: dict[str, bool] = {
        attr: col in playlists_df.columns for attr, col in soft_attr_columns.items()
    }

    n_with_xano_tags = 0
    n_with_text_tags = 0
    n_with_popularity = 0
    n_with_soft_attrs: dict[str, int] = {a: 0 for a in soft_attr_columns}

    for _, row in playlists_df.iterrows():
        pid = str(row["playlist_id"])
        text_emb = playlist_text_emb_by_id.get(
            pid, np.zeros(embedding_dim, dtype=np.float32)
        )

        accepted_track_ids = accepted_by_pl.get(pid, [])
        accepted_embs = [
            track_text_emb_by_id[t]
            for t in accepted_track_ids
            if t in track_text_emb_by_id
        ]
        if accepted_embs:
            accepted_centroid = np.mean(np.vstack(accepted_embs), axis=0)
            semantic = semantic_blend * text_emb + (1.0 - semantic_blend) * accepted_centroid
        else:
            semantic = text_emb
        semantic = _safe_normalize(semantic.astype(np.float32))

        accepted_audio = [
            audio_lookup[t] for t in accepted_track_ids if t in audio_lookup
        ]
        if accepted_audio:
            audio_arr = np.vstack(accepted_audio)
            with np.errstate(invalid="ignore"):
                audio_centroid = np.nanmean(audio_arr, axis=0)
                audio_std = np.nanstd(audio_arr, axis=0)
                audio_min = np.nanmin(audio_arr, axis=0)
                audio_max = np.nanmax(audio_arr, axis=0)
        else:
            audio_centroid = None
            audio_std = None
            audio_min = None
            audio_max = None

        accepted_pops = [
            popularity_lookup[t]
            for t in accepted_track_ids
            if t in popularity_lookup
        ]
        if accepted_pops:
            popularity_mean = float(np.mean(accepted_pops))
            popularity_std = float(np.std(accepted_pops))
            popularity_count = len(accepted_pops)
            n_with_popularity += 1
        else:
            popularity_mean = None
            popularity_std = None
            popularity_count = 0

        accepted_n = int(accepted_count_by_pl.get(pid, 0))
        declined_n = int(declined_count_by_pl.get(pid, 0))
        total = accepted_n + declined_n
        rate = accepted_n / total if total else 0.0

        # Tags: prefer authoritative Xano genres/subgenres, then add anything
        # the regex tagger picks up from the title/description. We also keep
        # the "primary" subset (canonical tags coming exclusively from the
        # `genres` array) so the explicit-genre filter can distinguish a
        # playlist that *is* a blues playlist from a rock playlist that
        # happens to include "Blues Rock" as a subgenre.
        primary_xano_tags = (
            normalize_xano_labels(
                _coerce_label_list(row.get("genres")) if has_xano_genres else None,
                None,
            )
            if has_xano_genres
            else set()
        )
        subgenre_xano_tags = (
            normalize_xano_labels(
                None,
                _coerce_label_list(row.get("subgenres")) if has_xano_subgenres else None,
            )
            if has_xano_subgenres
            else set()
        )
        xano_tags = primary_xano_tags | subgenre_xano_tags
        playlist_text = _playlist_text(row)
        text_tags = tag_text(playlist_text)
        tags = xano_tags | text_tags
        if xano_tags:
            n_with_xano_tags += 1
        if text_tags:
            n_with_text_tags += 1

        # Soft attributes — normalized lowercase sets, with "any"/"other"/
        # null already filtered out. Empty set => no preference.
        soft_attrs: dict[str, set[str]] = {}
        for attr_name, raw_col in soft_attr_columns.items():
            if not soft_attr_present[attr_name]:
                soft_attrs[attr_name] = set()
                continue
            raw_values = _coerce_label_list(row.get(raw_col))
            normalized = normalize_attribute_labels(raw_values)
            soft_attrs[attr_name] = normalized
            if normalized:
                n_with_soft_attrs[attr_name] += 1

        profiles[pid] = PlaylistProfile(
            playlist_id=pid,
            playlist_name=str(row.get("playlist_name") or ""),
            text_emb=text_emb.astype(np.float32),
            semantic_centroid=semantic,
            accepted_count=accepted_n,
            declined_count=declined_n,
            acceptance_rate=rate,
            audio_centroid=audio_centroid,
            audio_std=audio_std,
            audio_min=audio_min,
            audio_max=audio_max,
            tags=tags,
            primary_tags=primary_xano_tags,
            popularity_mean=popularity_mean,
            popularity_std=popularity_std,
            popularity_count=popularity_count,
            activities=soft_attrs["activities"],
            countries=soft_attrs["countries"],
            languages=soft_attrs["languages"],
            tempos=soft_attrs["tempos"],
            moods=soft_attrs["moods"],
            tier=coerce_playlist_tier(row.get("tier")),
        )

    soft_attr_summary = " ".join(
        f"{name}={n_with_soft_attrs[name]}" for name in soft_attr_columns
    )
    print(
        f"[Profiles] Built {len(profiles)} profiles | "
        f"with_audio_centroid={sum(1 for p in profiles.values() if p.audio_centroid is not None)} | "
        f"with_tags={sum(1 for p in profiles.values() if p.tags)} | "
        f"xano_tagged={n_with_xano_tags} | text_tagged={n_with_text_tags} | "
        f"with_popularity_stats={n_with_popularity} | "
        f"soft_attrs[{soft_attr_summary}]"
    )
    return ProfileBundle(
        profiles=profiles,
        audio_feature_cols=list(available_audio_cols),
        embedding_dim=embedding_dim,
    )
