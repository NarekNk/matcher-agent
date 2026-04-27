from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from matcher_agent.embeddings import TextEmbedder
from matcher_agent.features.feature_builder import (
    build_pair_features,
    build_track_audio_lookup,
    build_track_meta_lookup,
)
from matcher_agent.features.playlist_profiles import (
    AUDIO_FEATURE_COLS,
    ProfileBundle,
    build_playlist_text_strings,
    build_profiles,
    build_track_text_strings,
)


@dataclass
class TrainingDataBundle:
    pair_features: pd.DataFrame
    profile_bundle: ProfileBundle
    track_text_emb_by_id: dict[str, np.ndarray]
    playlist_text_emb_by_id: dict[str, np.ndarray]
    track_audio_by_id: dict[str, np.ndarray]
    track_meta_by_id: dict[str, dict]


def _embed_lookup(
    df: pd.DataFrame,
    *,
    id_col: str,
    text_strings: list[str],
    embedder: TextEmbedder,
) -> dict[str, np.ndarray]:
    embeddings = embedder.encode(text_strings)
    out: dict[str, np.ndarray] = {}
    for emb, raw_id in zip(embeddings, df[id_col].astype("string").tolist()):
        if raw_id and raw_id != "<NA>" and raw_id != "nan":
            out[str(raw_id)] = emb.astype(np.float32)
    return out


def build_training_bundle(
    matches_df: pd.DataFrame,
    tracks_df: pd.DataFrame,
    playlists_df: pd.DataFrame,
    *,
    text_embedder: TextEmbedder,
    semantic_blend: float = 0.5,
) -> TrainingDataBundle:
    print(
        f"[Dataset] Building bundle | matches={len(matches_df)} "
        f"tracks={len(tracks_df)} playlists={len(playlists_df)}"
    )

    matches_df = matches_df.copy()
    matches_df["track_id"] = matches_df["track_id"].astype("string")
    matches_df["playlist_id"] = matches_df["playlist_id"].astype("string")
    matches_df = matches_df.dropna(subset=["track_id", "playlist_id", "label"])

    track_meta_columns = [
        c for c in ("track_id", "track_name", "artist", "album", *AUDIO_FEATURE_COLS)
        if c in tracks_df.columns
    ]
    tracks_df = tracks_df[track_meta_columns].copy()
    tracks_df["track_id"] = tracks_df["track_id"].astype("string")
    tracks_df = tracks_df.drop_duplicates(subset=["track_id"], keep="last")

    # Hydrate track metadata for every track that appears in matches but is
    # missing from the audio export. Without this, text embeddings would only
    # cover tracks with audio previews; we want broad coverage.
    matches_meta = matches_df[
        [c for c in ("track_id", "track_name", "artist") if c in matches_df.columns]
    ].drop_duplicates(subset=["track_id"], keep="last")
    extras = matches_meta[~matches_meta["track_id"].isin(tracks_df["track_id"])]
    if not extras.empty:
        print(f"[Dataset] Adding {len(extras)} tracks without audio for text-only embeddings.")
        for col in tracks_df.columns:
            if col not in extras.columns:
                extras[col] = pd.NA
        tracks_df = pd.concat([tracks_df, extras[tracks_df.columns]], ignore_index=True)

    playlists_df = playlists_df.copy()
    playlists_df["playlist_id"] = playlists_df["playlist_id"].astype("string")
    playlists_df = playlists_df.drop_duplicates(subset=["playlist_id"], keep="last")

    print("[Dataset] Embedding playlist text.")
    playlist_text_strings = build_playlist_text_strings(playlists_df)
    playlist_text_emb_by_id = _embed_lookup(
        playlists_df,
        id_col="playlist_id",
        text_strings=playlist_text_strings,
        embedder=text_embedder,
    )

    print("[Dataset] Embedding track text.")
    track_text_strings = build_track_text_strings(tracks_df)
    track_text_emb_by_id = _embed_lookup(
        tracks_df,
        id_col="track_id",
        text_strings=track_text_strings,
        embedder=text_embedder,
    )

    profile_bundle = build_profiles(
        playlists_df,
        matches_df,
        tracks_df,
        track_text_emb_by_id=track_text_emb_by_id,
        playlist_text_emb_by_id=playlist_text_emb_by_id,
        semantic_blend=semantic_blend,
    )

    track_audio_by_id = build_track_audio_lookup(tracks_df, profile_bundle.audio_feature_cols)
    track_meta_by_id = build_track_meta_lookup(tracks_df)

    pair_features = build_pair_features(
        matches_df[["track_id", "playlist_id", "label"]],
        profile_bundle=profile_bundle,
        track_text_emb_by_id=track_text_emb_by_id,
        track_audio_by_id=track_audio_by_id,
        track_meta_by_id=track_meta_by_id,
    )

    return TrainingDataBundle(
        pair_features=pair_features,
        profile_bundle=profile_bundle,
        track_text_emb_by_id=track_text_emb_by_id,
        playlist_text_emb_by_id=playlist_text_emb_by_id,
        track_audio_by_id=track_audio_by_id,
        track_meta_by_id=track_meta_by_id,
    )
