from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from matcher_agent.artifacts.io import load_bundle
from matcher_agent.embeddings import TextEmbedder
from matcher_agent.features.feature_builder import (
    build_pair_features,
    build_track_audio_lookup,
    build_track_meta_lookup,
)
from matcher_agent.features.genre_normalizer import normalize_external_labels
from matcher_agent.features.genre_tagger import has_conflict, tag_text
from matcher_agent.features.playlist_profiles import (
    AUDIO_FEATURE_COLS,
    build_playlist_text_strings,
    build_profiles,
    build_track_popularity_lookup,
    build_track_text_strings,
)
from matcher_agent.models import PlaylistRecommendation, TrackInput


class MatcherService:
    """Online recommender. Profiles are computed once at construction time so
    each `recommend_playlists` call is fast (just embed the new track + score)."""

    def __init__(
        self,
        artifact_dir: str,
        historical_df: pd.DataFrame,
        playlists_df: pd.DataFrame,
        tracks_df: pd.DataFrame,
        text_embedder: TextEmbedder,
        *,
        semantic_blend: float = 0.5,
        hard_genre_filter: bool = True,
    ):
        print(f"[Recommend] Loading model artifacts from {artifact_dir}")
        bundle = load_bundle(Path(artifact_dir))
        self.model = bundle["model"]
        self.feature_columns = bundle["feature_columns"]
        self.text_embedder = text_embedder
        self.hard_genre_filter = hard_genre_filter

        playlists_df = playlists_df.copy()
        playlists_df["playlist_id"] = playlists_df["playlist_id"].astype("string")
        playlists_df = playlists_df.drop_duplicates(subset=["playlist_id"], keep="last")

        historical_df = historical_df.copy()
        historical_df["track_id"] = historical_df["track_id"].astype("string")
        historical_df["playlist_id"] = historical_df["playlist_id"].astype("string")
        historical_df = historical_df.dropna(subset=["track_id", "playlist_id", "label"])

        tracks_df = tracks_df.copy()
        tracks_df["track_id"] = tracks_df["track_id"].astype("string")
        tracks_df = tracks_df.drop_duplicates(subset=["track_id"], keep="last")

        # Hydrate any tracks that appear only in historical matches.
        match_meta = historical_df[
            [c for c in ("track_id", "track_name", "artist") if c in historical_df.columns]
        ].drop_duplicates(subset=["track_id"], keep="last")
        extras = match_meta[~match_meta["track_id"].isin(tracks_df["track_id"])]
        if not extras.empty:
            for col in tracks_df.columns:
                if col not in extras.columns:
                    extras[col] = pd.NA
            tracks_df = pd.concat([tracks_df, extras[tracks_df.columns]], ignore_index=True)

        print(
            f"[Recommend] Embedding {len(playlists_df)} playlists and {len(tracks_df)} tracks."
        )
        playlist_text_strings = build_playlist_text_strings(playlists_df)
        playlist_text_emb = self.text_embedder.encode(playlist_text_strings)
        track_text_strings = build_track_text_strings(tracks_df)
        track_text_emb = self.text_embedder.encode(track_text_strings)

        playlist_text_emb_by_id = {
            pid: emb.astype(np.float32)
            for pid, emb in zip(playlists_df["playlist_id"].astype(str), playlist_text_emb)
        }
        track_text_emb_by_id = {
            tid: emb.astype(np.float32)
            for tid, emb in zip(tracks_df["track_id"].astype(str), track_text_emb)
        }

        self.profile_bundle = build_profiles(
            playlists_df,
            historical_df,
            tracks_df,
            track_text_emb_by_id=track_text_emb_by_id,
            playlist_text_emb_by_id=playlist_text_emb_by_id,
            semantic_blend=semantic_blend,
        )
        self.track_audio_by_id = build_track_audio_lookup(
            tracks_df, self.profile_bundle.audio_feature_cols
        )
        self.track_meta_by_id = build_track_meta_lookup(tracks_df)
        self.track_popularity_by_id = build_track_popularity_lookup(tracks_df)
        self.playlists_df = playlists_df
        self.historical_df = historical_df

        print(
            f"[Recommend] Service ready playlists={len(self.profile_bundle.profiles)} "
            f"audio_features={len(self.profile_bundle.audio_feature_cols)} "
            f"feature_cols={len(self.feature_columns)}"
        )

    def _track_text_for_input(self, track: TrackInput) -> str:
        artist = (track.artist or "").strip()
        name = (track.track_name or "").strip()
        base = f"{artist} - {name}" if (artist and name) else (name or artist)
        # Inject Spotify-provided artist genres into the embedded text so the
        # semantic vector carries genre information even when the title doesn't.
        if track.artist_genres:
            genre_phrase = ", ".join(track.artist_genres)
            return f"{base}. Genres: {genre_phrase}."
        return base

    def recommend_playlists(self, track: TrackInput, n: int) -> list[PlaylistRecommendation]:
        track_id = track.track_id or "__adhoc_track__"
        text = self._track_text_for_input(track)
        track_tags = tag_text(text)
        # Spotify-provided artist genres are authoritative; map them through
        # the explicit normalizer (which handles compound labels like
        # "west coast hip hop", "neo soul", "drum and bass") and merge in.
        if track.artist_genres:
            track_tags |= normalize_external_labels(track.artist_genres)
        print(
            f"[Recommend] Scoring track='{track.track_name}' artist='{track.artist}' "
            f"popularity={track.popularity} "
            f"tags={sorted(track_tags) if track_tags else '[]'} n={n}"
        )

        track_emb = self.text_embedder.encode([text])[0].astype(np.float32)
        merged_track_text_emb = dict(
            (pid, emb) for pid, emb in [(track_id, track_emb)]
        )

        # Splice in the inference-time track so the feature builder sees it.
        track_text_lookup = {**self._dummy_lookup_for_other_tracks(), **merged_track_text_emb}
        meta_lookup = {
            **self.track_meta_by_id,
            track_id: {
                "track_name": track.track_name or "",
                "artist": track.artist or "",
                "album": track.album or "",
                "_cached_tags": track_tags,
            },
        }
        audio_lookup = dict(self.track_audio_by_id)
        track_audio_vec = self._extract_audio_vector(track)
        if track_audio_vec is not None:
            audio_lookup[track_id] = track_audio_vec

        # Override the popularity lookup with the inference-time value so
        # the popularity-fit features see the freshly fetched score.
        popularity_lookup = dict(self.track_popularity_by_id)
        if track.popularity is not None:
            popularity_lookup[track_id] = float(track.popularity)

        playlist_ids = list(self.profile_bundle.profiles.keys())
        pair_input = pd.DataFrame(
            {"track_id": [track_id] * len(playlist_ids), "playlist_id": playlist_ids}
        )

        feats = build_pair_features(
            pair_input,
            profile_bundle=self.profile_bundle,
            track_text_emb_by_id=track_text_lookup,
            track_audio_by_id=audio_lookup,
            track_meta_by_id=meta_lookup,
            track_popularity_by_id=popularity_lookup,
        )

        X = feats[self.feature_columns].copy()
        probs = self.model.predict_proba(X)[:, 1]
        feats["acceptance_probability"] = probs

        if self.hard_genre_filter:
            mask = self._genre_filter_mask(feats, track_tags)
            feats.loc[mask, "acceptance_probability"] *= 0.05

        ranked = feats.sort_values("acceptance_probability", ascending=False).head(n)
        playlist_name_by_id = {
            str(pid): self.profile_bundle.profiles[str(pid)].playlist_name
            for pid in ranked["playlist_id"].tolist()
            if str(pid) in self.profile_bundle.profiles
        }

        results: list[PlaylistRecommendation] = []
        for idx, (_, row) in enumerate(ranked.iterrows()):
            pid = str(row["playlist_id"])
            results.append(
                PlaylistRecommendation(
                    playlist_id=pid,
                    playlist_name=playlist_name_by_id.get(pid, pid),
                    acceptance_probability=float(row["acceptance_probability"]),
                    rank=idx + 1,
                )
            )
        print(f"[Recommend] Returned {len(results)} recommendations.")
        return results

    def _dummy_lookup_for_other_tracks(self) -> dict[str, np.ndarray]:
        # We only need the new track's embedding to be present; other tracks
        # are looked up from the existing profiles, not at score time.
        return {}

    def _extract_audio_vector(self, track: TrackInput) -> np.ndarray | None:
        cols = self.profile_bundle.audio_feature_cols
        if not cols:
            return None
        if not track.extra:
            return None
        vec = np.array(
            [pd.to_numeric(track.extra.get(c), errors="coerce") for c in cols],
            dtype=np.float64,
        )
        if np.all(np.isnan(vec)):
            return None
        return vec

    def _genre_filter_mask(self, feats: pd.DataFrame, track_tags: set[str]) -> pd.Series:
        if not track_tags:
            return pd.Series(False, index=feats.index)
        masks = []
        for pid in feats["playlist_id"].astype(str).tolist():
            prof = self.profile_bundle.profiles.get(pid)
            masks.append(False if prof is None else has_conflict(track_tags, prof.tags))
        return pd.Series(masks, index=feats.index)
