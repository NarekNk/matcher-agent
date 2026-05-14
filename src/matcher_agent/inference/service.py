from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from matcher_agent.artifacts.io import load_bundle
from matcher_agent.embeddings import TextEmbedder
from matcher_agent.features.attribute_normalizer import normalize_attribute_labels
from matcher_agent.features.feature_builder import (
    build_pair_features,
    build_track_audio_lookup,
    build_track_meta_lookup,
)
from matcher_agent.features.genre_normalizer import (
    normalize_external_labels,
    normalize_xano_labels,
)
from matcher_agent.features.genre_tagger import has_conflict, tag_text
from matcher_agent.features.playlist_profiles import (
    AUDIO_FEATURE_COLS,
    build_playlist_text_strings,
    build_profiles,
    build_track_popularity_lookup,
    build_track_text,
    build_track_text_strings,
    ensure_audio_columns,
)
from matcher_agent.models import PlaylistRecommendation, TrackInput, parse_track_tier


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
        semantic_blend: float = 0.25,
        hard_genre_filter: bool = True,
        soft_attribute_penalty: float = 0.7,
        language_mismatch_penalty: float = 0.3,
        explicit_genre_no_match_penalty: float = 0.02,
        explicit_genre_untagged_penalty: float = 0.3,
        explicit_genre_subgenre_only_penalty: float = 0.4,
        explicit_genre_broadtag_threshold: int = 4,
    ):
        print(f"[Recommend] Loading model artifacts from {artifact_dir}")
        bundle = load_bundle(Path(artifact_dir))
        self.model = bundle["model"]
        self.feature_columns = bundle["feature_columns"]
        self.text_embedder = text_embedder
        self.hard_genre_filter = hard_genre_filter
        # Clamp into (0, 1]; 1.0 disables soft penalties.
        self.soft_attribute_penalty = max(1e-6, min(1.0, float(soft_attribute_penalty)))
        self.language_mismatch_penalty = max(
            1e-6, min(1.0, float(language_mismatch_penalty))
        )
        # When the user explicitly supplies track genres/subgenres we switch
        # from "conflict avoidance" to a stricter "positive overlap required"
        # filter. These two knobs control that behavior:
        #   * `explicit_genre_no_match_penalty` -- multiplier for playlists
        #     whose canonical Xano tags do not share ANY tag with the user-
        #     supplied genres (default 0.02 ~ effectively dropped).
        #   * `explicit_genre_untagged_penalty` -- multiplier for playlists
        #     that have no Xano tags at all (we cannot verify fit, so we
        #     down-weight but don't drop -- default 0.3).
        self.explicit_genre_no_match_penalty = max(
            1e-6, min(1.0, float(explicit_genre_no_match_penalty))
        )
        self.explicit_genre_untagged_penalty = max(
            1e-6, min(1.0, float(explicit_genre_untagged_penalty))
        )
        # Tier penalty for "matched only via a subgenre" (e.g. a Rock
        # playlist with subgenre 'Blues Rock' against a Blues track).
        self.explicit_genre_subgenre_only_penalty = max(
            1e-6, min(1.0, float(explicit_genre_subgenre_only_penalty))
        )
        # Over-tagging guard: playlists with more primary Xano genres than
        # this threshold get scaled down (curator selected the entire
        # genre dropdown -> generic catch-all -> not a real match).
        self.explicit_genre_broadtag_threshold = max(
            1, int(explicit_genre_broadtag_threshold)
        )

        playlists_df = playlists_df.copy()
        playlists_df["playlist_id"] = playlists_df["playlist_id"].astype("string")
        playlists_df = playlists_df.drop_duplicates(subset=["playlist_id"], keep="last")

        historical_df = historical_df.copy()
        historical_df["track_id"] = historical_df["track_id"].astype("string")
        historical_df["playlist_id"] = historical_df["playlist_id"].astype("string")
        historical_df = historical_df.dropna(subset=["track_id", "playlist_id", "label"])

        tracks_df = tracks_df.copy()
        tracks_df = ensure_audio_columns(tracks_df)
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
        """Build the text we embed for ``semantic_similarity``.

        Delegates to the shared ``build_track_text`` so training and
        inference produce identical text for the same metadata. All
        authoritative genre signals (Spotify artist_genres, user-supplied
        genres/subgenres) are injected so the text embedding is genre-aware.
        """
        lang_labels = sorted(normalize_attribute_labels(track.languages))
        mood_labels = sorted(normalize_attribute_labels(track.moods))
        return build_track_text(
            artist=track.artist or "",
            track_name=track.track_name or "",
            artist_genres=track.artist_genres or None,
            genres=track.genres or None,
            subgenres=track.subgenres or None,
            languages=lang_labels or None,
            moods=mood_labels or None,
        )

    def recommend_playlists(self, track: TrackInput, n: int) -> list[PlaylistRecommendation]:
        track_id = track.track_id or "__adhoc_track__"
        text = self._track_text_for_input(track)
        track_tags = tag_text(text)
        # Spotify-provided artist genres are authoritative; map them through
        # the explicit normalizer (which handles compound labels like
        # "west coast hip hop", "neo soul", "drum and bass") and merge in.
        spotify_track_tags: set[str] = set()
        if track.artist_genres:
            spotify_track_tags = normalize_external_labels(track.artist_genres)
            track_tags |= spotify_track_tags
        # User-supplied Xano-style track genres/subgenres (the same vocabulary
        # the curators use on playlists). These are the most authoritative
        # source when present, since the user is hand-classifying.
        user_explicit_track_tags: set[str] = set()
        if track.genres or track.subgenres:
            user_explicit_track_tags = normalize_xano_labels(
                track.genres, track.subgenres
            )
            track_tags |= user_explicit_track_tags

        # The "authoritative" tag set used by the strict positive-overlap
        # filter. Excludes tag_text(...) on artist/title because that often
        # picks up false positives from track names ("Country Song" by a
        # non-country artist). Includes Spotify artist_genres because those
        # are externally curated.
        authoritative_track_tags = user_explicit_track_tags | spotify_track_tags
        explicit_filter_active = (
            self.hard_genre_filter and bool(user_explicit_track_tags)
        )

        # Normalize user-supplied soft attributes (mood/language/etc.) once.
        # Empty-set sentinels mean "no preference"; they bypass the penalty.
        track_soft = {
            "activities": normalize_attribute_labels(track.activities),
            "countries": normalize_attribute_labels(track.countries),
            "languages": normalize_attribute_labels(track.languages),
            "tempos": normalize_attribute_labels(track.tempos),
            "moods": normalize_attribute_labels(track.moods),
        }
        soft_summary = {k: sorted(v) for k, v in track_soft.items() if v}
        track_tier = parse_track_tier(track.tier)
        print(
            f"[Recommend] Scoring track='{track.track_name}' artist='{track.artist}' "
            f"popularity={track.popularity} "
            f"tier={track_tier} "
            f"tags={sorted(track_tags) if track_tags else '[]'} "
            f"soft={soft_summary if soft_summary else '{}'} "
            f"n={n}"
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
                "_soft_moods": track_soft.get("moods", set()),
                "_soft_languages": track_soft.get("languages", set()),
                "_soft_activities": track_soft.get("activities", set()),
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
        if track_tier is not None:
            n_before = len(playlist_ids)
            playlist_ids = [
                pid
                for pid in playlist_ids
                if self.profile_bundle.profiles[pid].tier == track_tier
            ]
            print(
                f"[Recommend] Tier filter active (track_tier={track_tier}): "
                f"candidates {n_before} -> {len(playlist_ids)}"
            )
            if not playlist_ids:
                print("[Recommend] No playlists match track tier; returning [].")
                return []

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

        if explicit_filter_active:
            multipliers, counts = self._explicit_overlap_multipliers(
                feats, authoritative_track_tags
            )
            feats["acceptance_probability"] *= multipliers
            print(
                "[Recommend] Explicit-genre filter applied "
                f"(track_tags={sorted(authoritative_track_tags)} "
                f"no_match={self.explicit_genre_no_match_penalty:.2f} "
                f"subgenre_only={self.explicit_genre_subgenre_only_penalty:.2f} "
                f"untagged={self.explicit_genre_untagged_penalty:.2f} "
                f"broadtag_threshold={self.explicit_genre_broadtag_threshold}): "
                f"primary_overlap={counts['primary_overlap']} "
                f"mixed_primary={counts['mixed_primary_penalized']} "
                f"subgenre_only={counts['subgenre_only']} "
                f"no_overlap={counts['no_overlap']} "
                f"untagged_count={counts['untagged']} "
                f"broadtag_penalized={counts['broadtag_penalized']}"
            )
        elif self.hard_genre_filter:
            mask = self._genre_filter_mask(feats, track_tags)
            feats.loc[mask, "acceptance_probability"] *= 0.05

        soft_penalties_on = self.soft_attribute_penalty < 1.0 or (
            bool(track_soft.get("languages")) and self.language_mismatch_penalty < 1.0
        )
        if any(track_soft.values()) and soft_penalties_on:
            multipliers, total_conflicts = self._soft_attribute_multipliers(
                feats, track_soft
            )
            feats["acceptance_probability"] *= multipliers
            print(
                f"[Recommend] Soft-attribute penalty applied "
                f"(per-conflict={self.soft_attribute_penalty:.2f} "
                f"language_mismatch={self.language_mismatch_penalty:.2f}): "
                f"total_pair_conflicts={int(total_conflicts)}"
            )

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

    def _explicit_overlap_multipliers(
        self, feats: pd.DataFrame, authoritative_track_tags: set[str]
    ) -> tuple[pd.Series, dict[str, int]]:
        """Strict tiered positive-overlap filter with over-tagging guard.

        Active only when the user explicitly supplied track genres/subgenres.
        For each candidate playlist:
          1. Tier multiplier:
             * user tags overlap the playlist's *primary* Xano genres
               -> 1.0 (genuine genre match), **unless** other primaries remain
               that ``has_conflict`` with the user's tags (e.g. Hip-Hop + Pop
               on a trap playlist vs a pop-only user) -> ``subgenre_only_penalty``
             * user tags overlap any playlist tag (subgenre or text-derived)
               but no primary overlap -> `subgenre_only_penalty` (e.g. 0.4)
             * playlist has no tags at all -> `untagged_penalty` (e.g. 0.3)
             * playlist has tags but zero overlap -> `no_match_penalty`
               (e.g. 0.02 -- effectively dropped)
          2. Breadth multiplier: if the playlist's primary genre count
             exceeds `broadtag_threshold`, apply a **sqrt** decay:
             ``sqrt(threshold / len(primary_tags))``.  This is gentler
             than the previous linear ``threshold / len(...)`` rule —
             a playlist with 12 primary tags and threshold=4 now gets
             ``sqrt(4/12)≈0.58`` instead of ``4/12≈0.33``.  This
             avoids over-penalizing playlists that are legitimately
             multi-genre while still down-weighting "tag-everything"
             catch-all playlists.
        Final = tier * breadth.

        Without (1), a Rock playlist with "Blues Rock" subgenre wins
        against a real Blues primary playlist on `semantic_similarity`.
        Without (2), a "tag-everything" lo-fi/chillhop playlist matches
        every track because the curator selected all 24 primary genres.
        """
        import math

        n = len(feats)
        multipliers = np.ones(n, dtype=np.float64)
        counts: dict[str, int] = {
            "primary_overlap": 0,
            "subgenre_only": 0,
            "no_overlap": 0,
            "untagged": 0,
            "broadtag_penalized": 0,
            "mixed_primary_penalized": 0,
        }
        if not authoritative_track_tags:
            return pd.Series(multipliers, index=feats.index), counts

        threshold = self.explicit_genre_broadtag_threshold
        playlist_ids = feats["playlist_id"].astype(str).tolist()
        for i, pid in enumerate(playlist_ids):
            prof = self.profile_bundle.profiles.get(pid)
            if prof is None or not prof.tags:
                multipliers[i] = self.explicit_genre_untagged_penalty
                counts["untagged"] += 1
                continue
            primary = prof.primary_tags
            if primary and (authoritative_track_tags & primary):
                tier_mult = 1.0
                counts["primary_overlap"] += 1
                # Curators often tick both Hip-Hop and Pop on trap playlists.
                # Primary overlap on ``pop`` alone must not grant a full pass
                # when other primaries conflict with the user's explicit tags.
                primary_rest = primary - authoritative_track_tags
                if primary_rest and has_conflict(authoritative_track_tags, primary_rest):
                    tier_mult = self.explicit_genre_subgenre_only_penalty
                    counts["mixed_primary_penalized"] += 1
            elif authoritative_track_tags & prof.tags:
                tier_mult = self.explicit_genre_subgenre_only_penalty
                counts["subgenre_only"] += 1
            else:
                tier_mult = self.explicit_genre_no_match_penalty
                counts["no_overlap"] += 1
            # Over-tagging guard: sqrt-based decay instead of linear.
            breadth = len(primary) if primary else len(prof.tags)
            if breadth > threshold:
                breadth_mult = math.sqrt(float(threshold) / float(breadth))
                counts["broadtag_penalized"] += 1
            else:
                breadth_mult = 1.0
            multipliers[i] = tier_mult * breadth_mult
        return pd.Series(multipliers, index=feats.index), counts

    def _soft_attribute_multipliers(
        self, feats: pd.DataFrame, track_soft: dict[str, set[str]]
    ) -> tuple[pd.Series, int]:
        """Return per-row score multipliers from the soft-attribute penalty.

        For each candidate playlist, count attributes where:
          * the user provided a non-empty track-side set, AND
          * the playlist has a non-empty curator-set, AND
          * those two sets are disjoint (no overlap).
        Language mismatches use ``self.language_mismatch_penalty`` (typically
        stricter than ``self.soft_attribute_penalty``). Other soft attributes
        use ``self.soft_attribute_penalty``. Per-row multiplier is the product
        of the applicable penalties for each conflicting attribute.

        Returns the multiplier series and the total conflict count across
        all rows (logged for diagnostics).
        """
        active_attrs = [
            (name, values) for name, values in track_soft.items() if values
        ]
        if not active_attrs:
            return pd.Series(1.0, index=feats.index), 0

        conflicts = np.zeros(len(feats), dtype=np.int32)
        multipliers_arr = np.ones(len(feats), dtype=np.float64)
        playlist_ids = feats["playlist_id"].astype(str).tolist()
        for i, pid in enumerate(playlist_ids):
            prof = self.profile_bundle.profiles.get(pid)
            if prof is None:
                continue
            pl_attrs = prof.soft_attribute_sets()
            m = 1.0
            for name, track_values in active_attrs:
                pl_values = pl_attrs.get(name, set())
                if pl_values and not (track_values & pl_values):
                    conflicts[i] += 1
                    if name == "languages":
                        m *= self.language_mismatch_penalty
                    else:
                        m *= self.soft_attribute_penalty
            multipliers_arr[i] = m
        return pd.Series(multipliers_arr, index=feats.index), int(conflicts.sum())
