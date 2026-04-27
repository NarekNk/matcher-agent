from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from matcher_agent.features.genre_tagger import tag_text

# Audio columns used to build per-playlist accepted-track centroids. Kept small
# and meaningful so missing audio for some tracks doesn't dominate noise.
AUDIO_FEATURE_COLS: tuple[str, ...] = (
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
)


def _track_text(row: pd.Series) -> str:
    artist = str(row.get("artist") or "").strip()
    name = str(row.get("track_name") or "").strip()
    if artist and name:
        return f"{artist} - {name}"
    return name or artist


def _playlist_text(row: pd.Series) -> str:
    name = str(row.get("playlist_name") or "").strip()
    desc = str(row.get("description") or "").strip()
    if desc and desc.lower() != "nan":
        return f"{name}. {desc}"
    return name


def _safe_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        return vec
    return vec / norm


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
                    "playlist_tags": prof.tags,
                }
            )
        return pd.DataFrame(rows)


def build_playlist_text_strings(playlists_df: pd.DataFrame) -> list[str]:
    return [_playlist_text(row) for _, row in playlists_df.iterrows()]


def build_track_text_strings(tracks_df: pd.DataFrame) -> list[str]:
    return [_track_text(row) for _, row in tracks_df.iterrows()]


def build_profiles(
    playlists_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    tracks_df: pd.DataFrame,
    *,
    track_text_emb_by_id: dict[str, np.ndarray],
    playlist_text_emb_by_id: dict[str, np.ndarray],
    label_col: str = "label",
    audio_feature_cols: tuple[str, ...] = AUDIO_FEATURE_COLS,
    semantic_blend: float = 0.5,
) -> ProfileBundle:
    """Compute one PlaylistProfile per playlist in `playlists_df`.

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

    profiles: dict[str, PlaylistProfile] = {}
    embedding_dim = next(iter(playlist_text_emb_by_id.values())).shape[0] if playlist_text_emb_by_id else 384

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
        else:
            audio_centroid = None
            audio_std = None

        accepted_n = int(accepted_count_by_pl.get(pid, 0))
        declined_n = int(declined_count_by_pl.get(pid, 0))
        total = accepted_n + declined_n
        rate = accepted_n / total if total else 0.0

        playlist_text = _playlist_text(row)
        tags = tag_text(playlist_text)

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
            tags=tags,
        )

    print(
        f"[Profiles] Built {len(profiles)} profiles | "
        f"with_audio_centroid={sum(1 for p in profiles.values() if p.audio_centroid is not None)} | "
        f"with_tags={sum(1 for p in profiles.values() if p.tags)}"
    )
    return ProfileBundle(
        profiles=profiles,
        audio_feature_cols=list(available_audio_cols),
        embedding_dim=embedding_dim,
    )
