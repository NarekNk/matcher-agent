from __future__ import annotations

import numpy as np
import pandas as pd

from matcher_agent.features.genre_tagger import (
    has_conflict,
    jaccard,
    tag_text,
)
from matcher_agent.features.playlist_profiles import (
    AUDIO_FEATURE_COLS,
    ProfileBundle,
    _safe_normalize,
)

# These feature names are the contract between training, persistence, and
# inference. Adding/removing a name requires retraining.
#
# NOTE: Playlist-prior features (acceptance_rate, accepted_count,
# declined_count) are intentionally EXCLUDED from the model. In our data
# they're popularity confounders that, if included, dominate the model and
# cause it to recommend the same handful of high-acceptance playlists for
# every track regardless of genre. They are still computed and exposed for
# downstream re-ranking, just not fed to the GBM.
#
# NOTE: `genre_tag_count_track` and `genre_tag_count_playlist` are also
# excluded from the model. They were Top-4 in importance only because they
# acted as a proxy for "is this row Xano-tagged with a rich genre profile",
# which is a data-completeness signal, not a genre-fit signal. The real
# genre fit lives in `genre_jaccard`, `genre_overlap_count`,
# `genre_conflict_flag`, and playlist-side `genre_precision_*` — which only
# become discriminative once we add genre-conflict hard negatives during
# training. We still COMPUTE the count columns in `build_pair_features` so older
# saved models that include them
# in their `feature_columns` still load and infer correctly.
PAIRWISE_FEATURE_COLS: list[str] = [
    "semantic_similarity",
    "title_text_similarity",
    "audio_centroid_cosine",
    "audio_centroid_l2",
    "audio_zscore_mean",
    "audio_zscore_max",
    "genre_jaccard",
    "genre_overlap_count",
    # Playlist-side precision: overlap / |playlist primary genres| (fallback to
    # full tag set when primary is empty). Rewards 1/1 over 1/15 primary hits.
    "genre_precision_primary",
    "genre_precision_all",
    "genre_conflict_flag",
    "track_audio_available",
    # Track popularity vs playlist-accepted-popularity. These let the model
    # learn the "low-popularity playlists prefer low-popularity tracks"
    # pattern without leaking absolute popularity (which would bias toward
    # only-mainstream playlists). Track popularity itself is also exposed,
    # since some playlists strongly skew one way.
    "track_popularity_norm",
    "popularity_diff_norm",
    "popularity_zscore",
    "popularity_available",
    # --- Interaction features ---
    # Amplify the signal when both genre-fit and semantic-fit agree (or both
    # disagree). Tree models can approximate interactions, but giving them
    # the product explicitly makes the split much cheaper to learn.
    "genre_semantic_interaction",
    "audio_genre_interaction",
    # --- Text embedding features beyond cosine ---
    "semantic_l2_distance",
    # Divergence between how well the track fits the playlist's *accepted
    # tracks* centroid vs its title/description alone. High values indicate
    # the playlist's historical preferences differ from its stated genre.
    "title_semantic_diff",
    # --- Per-dimension audio diffs (z-score) for the most discriminative
    # audio attributes. Complement the aggregate zscore_mean/max with
    # fine-grained dimension signals. Default to 0.0 when the dimension is
    # not in the audio feature set or audio data is missing.
    "bpm_diff",
    "energy_diff",
    "danceability_diff",
    # Fraction of audio dimensions where the track's value falls within the
    # observed [min, max] range of the playlist's accepted tracks. Captures
    # the "audio envelope" fit better than centroid distance alone.
    "audio_range_ratio",
    # --- Playlist-level structural features ---
    # log-compressed to avoid the dominance issue noted for the raw counts.
    "playlist_size_log",
    "playlist_genre_richness",
    "playlist_audio_available",
    # --- Soft-attribute features ---
    # Playlist-side structural: always available, lets the model learn that
    # attribute-rich playlists behave differently from attribute-sparse ones.
    "playlist_n_soft_attrs",
    "playlist_has_language",
    "playlist_has_mood",
    # Pairwise overlap: requires track-side soft attributes which are only
    # available at inference time (user-supplied via TrackInput). At training
    # time these default to 0.0 with soft_attr_available=0.0 so the model
    # learns to ignore them when data is missing. As track-side attributes
    # enter the training set in the future, these features will naturally
    # activate and the model will learn proper weights.
    "mood_match_flag",
    "language_match_flag",
    "activity_match_flag",
    "soft_attr_available",
]


def _norm(text: str | None) -> str:
    return (text or "").strip().lower()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _audio_zscore_diff(
    track_audio: np.ndarray, centroid: np.ndarray, std: np.ndarray
) -> tuple[float, float]:
    if track_audio is None or centroid is None or std is None:
        return 0.0, 0.0
    diff = np.abs(track_audio - centroid) / (std + 1e-3)
    diff = diff[~np.isnan(diff)]
    if diff.size == 0:
        return 0.0, 0.0
    return float(np.mean(diff)), float(np.max(diff))


def _audio_l2(track_audio: np.ndarray, centroid: np.ndarray) -> float:
    if track_audio is None or centroid is None:
        return 0.0
    diff = track_audio - centroid
    diff = diff[~np.isnan(diff)]
    if diff.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(diff * diff)))


def _l2(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """L2 (Euclidean) distance between two vectors. 0.0 when either is None."""
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    diff = diff[~np.isnan(diff)]
    if diff.size == 0:
        return 0.0
    return float(np.sqrt(np.dot(diff, diff)))


def _per_dim_zscore(
    track_audio: np.ndarray | None,
    centroid: np.ndarray | None,
    std: np.ndarray | None,
    dim_index: int | None,
) -> float:
    """Z-score for a single audio dimension. 0.0 when data is missing."""
    if dim_index is None or track_audio is None or centroid is None or std is None:
        return 0.0
    if dim_index >= len(track_audio):
        return 0.0
    tv = track_audio[dim_index]
    cv = centroid[dim_index]
    sv = std[dim_index]
    if np.isnan(tv) or np.isnan(cv):
        return 0.0
    return float(abs(tv - cv) / (sv + 1e-3))


def _audio_range_ratio(
    track_audio: np.ndarray | None,
    audio_min: np.ndarray | None,
    audio_max: np.ndarray | None,
) -> float:
    """Fraction of audio dims where track value falls within [min, max]."""
    if track_audio is None or audio_min is None or audio_max is None:
        return 0.0
    valid = 0
    inside = 0
    for i in range(len(track_audio)):
        tv = track_audio[i]
        lo = audio_min[i]
        hi = audio_max[i]
        if np.isnan(tv) or np.isnan(lo) or np.isnan(hi):
            continue
        valid += 1
        if lo <= tv <= hi:
            inside += 1
    if valid == 0:
        return 0.0
    return float(inside / valid)


def _popularity_features(
    track_pop: float | None,
    playlist_mean: float | None,
    playlist_std: float | None,
) -> dict[str, float]:
    """Compute the popularity-fit features for a (track, playlist) pair.

    Conventions:
      * `track_popularity_norm` is in [0, 1]; 0 when the track has no popularity.
      * `popularity_diff_norm` is |track - playlist_mean| / 100, in [0, 1];
        0 when either side is missing.
      * `popularity_zscore` uses the playlist's std (with a 5-point floor so
        playlists with only 1-2 accepted tracks don't blow up). 0 when missing.
      * `popularity_available` is 1.0 only when BOTH sides are available, so
        the model can learn to trust the diff signal only when it's real.
    """
    if track_pop is None or np.isnan(track_pop):
        track_norm = 0.0
        track_known = False
    else:
        track_norm = max(0.0, min(1.0, float(track_pop) / 100.0))
        track_known = True

    if (
        playlist_mean is None
        or np.isnan(playlist_mean)
        or not track_known
    ):
        diff_norm = 0.0
        zscore = 0.0
        available = 0.0
    else:
        diff = abs(float(track_pop) - float(playlist_mean))
        diff_norm = max(0.0, min(1.0, diff / 100.0))
        std = float(playlist_std) if playlist_std is not None and not np.isnan(playlist_std) else 0.0
        zscore = diff / max(std, 5.0)
        available = 1.0
    return {
        "track_popularity_norm": track_norm,
        "popularity_diff_norm": diff_norm,
        "popularity_zscore": float(zscore),
        "popularity_available": available,
    }


def build_track_audio_lookup(
    tracks_df: pd.DataFrame, audio_feature_cols: list[str]
) -> dict[str, np.ndarray]:
    if not audio_feature_cols:
        return {}
    cols = ["track_id", *[c for c in audio_feature_cols if c in tracks_df.columns]]
    if len(cols) == 1:
        return {}
    sub = tracks_df[cols].copy()
    sub["track_id"] = sub["track_id"].astype("string")
    out: dict[str, np.ndarray] = {}
    audio_cols = cols[1:]
    for _, row in sub.iterrows():
        tid = row["track_id"]
        if pd.isna(tid):
            continue
        vec = np.array(
            [pd.to_numeric(row[c], errors="coerce") for c in audio_cols],
            dtype=np.float64,
        )
        if not np.all(np.isnan(vec)):
            out[str(tid)] = vec
    return out


def build_pair_features(
    pairs_df: pd.DataFrame,
    *,
    profile_bundle: ProfileBundle,
    track_text_emb_by_id: dict[str, np.ndarray],
    track_audio_by_id: dict[str, np.ndarray],
    track_meta_by_id: dict[str, dict],
    track_popularity_by_id: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute pairwise features for every (track_id, playlist_id) row.

    `pairs_df` must include track_id, playlist_id columns. Other columns
    (label, etc.) are passed through.
    """
    print(f"[Features] Computing pairwise features for {len(pairs_df)} rows.")

    rows = pairs_df.copy()
    rows["track_id"] = rows["track_id"].astype("string")
    rows["playlist_id"] = rows["playlist_id"].astype("string")

    profiles = profile_bundle.profiles
    audio_cols = profile_bundle.audio_feature_cols
    audio_dim = len(audio_cols)
    zero_audio = np.full(audio_dim, np.nan, dtype=np.float64) if audio_dim else None
    pop_lookup = track_popularity_by_id or {}

    _audio_idx: dict[str, int] = {name: i for i, name in enumerate(audio_cols)}
    bpm_idx = _audio_idx.get("bpm")
    energy_idx = _audio_idx.get("energy")
    dance_idx = _audio_idx.get("danceability")

    out_records: list[dict] = []
    for record in rows.to_dict(orient="records"):
        tid = str(record["track_id"])
        pid = str(record["playlist_id"])
        prof = profiles.get(pid)
        meta = track_meta_by_id.get(tid, {})
        track_emb = track_text_emb_by_id.get(tid)
        track_audio = track_audio_by_id.get(tid)
        track_text = (
            f"{meta.get('artist','')} {meta.get('track_name','')} {meta.get('album','')}"
        ).strip()
        track_tags = meta.get("_cached_tags") or tag_text(track_text)
        meta["_cached_tags"] = track_tags

        if prof is not None and track_emb is not None:
            semantic_similarity = _cosine(track_emb, prof.semantic_centroid)
            title_text_similarity = _cosine(track_emb, prof.text_emb)
            semantic_l2_distance = _l2(track_emb, prof.semantic_centroid)
        else:
            semantic_similarity = 0.0
            title_text_similarity = 0.0
            semantic_l2_distance = 0.0

        title_semantic_diff = abs(semantic_similarity - title_text_similarity)

        if prof is not None and prof.audio_centroid is not None and track_audio is not None:
            audio_norm_track = _safe_normalize(track_audio.astype(np.float64))
            audio_norm_centroid = _safe_normalize(prof.audio_centroid.astype(np.float64))
            audio_centroid_cosine = _cosine(audio_norm_track, audio_norm_centroid)
            audio_centroid_l2 = _audio_l2(track_audio, prof.audio_centroid)
            audio_z_mean, audio_z_max = _audio_zscore_diff(
                track_audio, prof.audio_centroid, prof.audio_std
            )
            track_audio_available = 1.0
            bpm_diff_val = _per_dim_zscore(track_audio, prof.audio_centroid, prof.audio_std, bpm_idx)
            energy_diff_val = _per_dim_zscore(track_audio, prof.audio_centroid, prof.audio_std, energy_idx)
            dance_diff_val = _per_dim_zscore(track_audio, prof.audio_centroid, prof.audio_std, dance_idx)
            audio_range_ratio = _audio_range_ratio(track_audio, prof.audio_min, prof.audio_max)
        else:
            audio_centroid_cosine = 0.0
            audio_centroid_l2 = 0.0
            audio_z_mean = 0.0
            audio_z_max = 0.0
            track_audio_available = 1.0 if track_audio is not None else 0.0
            bpm_diff_val = 0.0
            energy_diff_val = 0.0
            dance_diff_val = 0.0
            audio_range_ratio = 0.0

        if prof is not None:
            tags_overlap = jaccard(track_tags, prof.tags)
            overlap_count = float(len(track_tags & prof.tags))
            conflict = 1.0 if has_conflict(track_tags, prof.tags) else 0.0
            primary = prof.primary_tags
            all_pl_tags = prof.tags
            if primary:
                genre_precision_primary = len(track_tags & primary) / max(
                    len(primary), 1
                )
            elif all_pl_tags:
                genre_precision_primary = len(track_tags & all_pl_tags) / max(
                    len(all_pl_tags), 1
                )
            else:
                genre_precision_primary = 0.0
            if all_pl_tags:
                genre_precision_all = len(track_tags & all_pl_tags) / max(
                    len(all_pl_tags), 1
                )
            else:
                genre_precision_all = 0.0
            tag_count_pl = float(len(prof.tags))
            acceptance_rate = float(prof.acceptance_rate)
            accepted_count = float(prof.accepted_count)
            declined_count = float(prof.declined_count)
            playlist_pop_mean = prof.popularity_mean
            playlist_pop_std = prof.popularity_std
            playlist_size_log = float(np.log1p(prof.accepted_count))
            playlist_genre_richness = float(np.log1p(len(prof.tags)))
            playlist_audio_available = 1.0 if prof.audio_centroid is not None else 0.0
            pl_soft = prof.soft_attribute_sets()
            playlist_n_soft_attrs = float(sum(1 for v in pl_soft.values() if v))
            playlist_has_language = 1.0 if pl_soft.get("languages") else 0.0
            playlist_has_mood = 1.0 if pl_soft.get("moods") else 0.0
        else:
            tags_overlap = 0.0
            overlap_count = 0.0
            conflict = 0.0
            genre_precision_primary = 0.0
            genre_precision_all = 0.0
            tag_count_pl = 0.0
            acceptance_rate = 0.0
            accepted_count = 0.0
            declined_count = 0.0
            playlist_pop_mean = None
            playlist_pop_std = None
            playlist_size_log = 0.0
            playlist_genre_richness = 0.0
            playlist_audio_available = 0.0
            pl_soft = {}
            playlist_n_soft_attrs = 0.0
            playlist_has_language = 0.0
            playlist_has_mood = 0.0

        # Pairwise soft-attribute overlap features. Track-side soft attrs
        # are injected via `_soft_*` keys in the track_meta dict at inference
        # time. At training time these keys are absent, so all overlap
        # features default to 0.0 with soft_attr_available=0.0.
        track_soft_moods: set[str] = meta.get("_soft_moods", set())
        track_soft_langs: set[str] = meta.get("_soft_languages", set())
        track_soft_acts: set[str] = meta.get("_soft_activities", set())
        has_any_soft = bool(track_soft_moods or track_soft_langs or track_soft_acts)
        soft_attr_available = 1.0 if has_any_soft else 0.0

        pl_moods = pl_soft.get("moods", set())
        pl_langs = pl_soft.get("languages", set())
        pl_acts = pl_soft.get("activities", set())
        mood_match_flag = (
            1.0 if (track_soft_moods and pl_moods and (track_soft_moods & pl_moods)) else 0.0
        )
        language_match_flag = (
            1.0 if (track_soft_langs and pl_langs and (track_soft_langs & pl_langs)) else 0.0
        )
        if track_soft_acts and pl_acts:
            union = len(track_soft_acts | pl_acts)
            activity_match_flag = float(len(track_soft_acts & pl_acts)) / union if union else 0.0
        else:
            activity_match_flag = 0.0

        pop_feats = _popularity_features(
            pop_lookup.get(tid),
            playlist_pop_mean,
            playlist_pop_std,
        )

        out_records.append(
            {
                **record,
                "semantic_similarity": semantic_similarity,
                "title_text_similarity": title_text_similarity,
                "audio_centroid_cosine": audio_centroid_cosine,
                "audio_centroid_l2": audio_centroid_l2,
                "audio_zscore_mean": audio_z_mean,
                "audio_zscore_max": audio_z_max,
                "genre_jaccard": tags_overlap,
                "genre_overlap_count": overlap_count,
                "genre_precision_primary": float(genre_precision_primary),
                "genre_precision_all": float(genre_precision_all),
                "genre_conflict_flag": conflict,
                "genre_tag_count_track": float(len(track_tags)),
                "genre_tag_count_playlist": tag_count_pl,
                "playlist_acceptance_rate": acceptance_rate,
                "playlist_accepted_track_count": accepted_count,
                "playlist_declined_track_count": declined_count,
                "track_audio_available": track_audio_available,
                **pop_feats,
                # --- New features ---
                "genre_semantic_interaction": tags_overlap * semantic_similarity,
                "audio_genre_interaction": audio_centroid_cosine * overlap_count,
                "semantic_l2_distance": semantic_l2_distance,
                "title_semantic_diff": title_semantic_diff,
                "bpm_diff": bpm_diff_val,
                "energy_diff": energy_diff_val,
                "danceability_diff": dance_diff_val,
                "audio_range_ratio": audio_range_ratio,
                "playlist_size_log": playlist_size_log,
                "playlist_genre_richness": playlist_genre_richness,
                "playlist_audio_available": playlist_audio_available,
                "playlist_n_soft_attrs": playlist_n_soft_attrs,
                "playlist_has_language": playlist_has_language,
                "playlist_has_mood": playlist_has_mood,
                "mood_match_flag": mood_match_flag,
                "language_match_flag": language_match_flag,
                "activity_match_flag": activity_match_flag,
                "soft_attr_available": soft_attr_available,
            }
        )

    df = pd.DataFrame(out_records)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def select_model_features(
    df: pd.DataFrame, *, excluded: set[str] | None = None
) -> list[str]:
    """Return the explicit, fixed pairwise feature column list, in fixed order.

    We deliberately do NOT auto-discover columns: the previous implementation
    pulled in identifier-like fields and prior-only signals that caused
    leakage and prevented learning track↔playlist fit. Anything that needs to
    be a feature must be added to PAIRWISE_FEATURE_COLS explicitly.
    """
    excluded = excluded or set()
    return [c for c in PAIRWISE_FEATURE_COLS if c in df.columns and c not in excluded]


def build_track_meta_lookup(tracks_df: pd.DataFrame) -> dict[str, dict]:
    """Per-track metadata bag used by the pairwise feature builder."""
    meta_cols = [c for c in ("track_name", "artist", "album") if c in tracks_df.columns]
    sub = tracks_df[["track_id", *meta_cols]].copy()
    sub["track_id"] = sub["track_id"].astype("string")
    out: dict[str, dict] = {}
    for _, row in sub.iterrows():
        tid = row["track_id"]
        if pd.isna(tid):
            continue
        out[str(tid)] = {c: ("" if pd.isna(row[c]) else str(row[c])) for c in meta_cols}
    return out
