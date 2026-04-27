from __future__ import annotations

import pandas as pd


def generate_candidates(
    playlists_df: pd.DataFrame, *, limit: int | None = None
) -> pd.DataFrame:
    """Return the candidate playlists to score. Today we score every catalog
    playlist; the GBM ranker is cheap once features are precomputed. If the
    catalog grows past ~50k, swap this for an embedding-based top-K retrieval
    over `playlist_semantic_centroid` and pass the surviving ids in.
    """
    cols = [
        c
        for c in (
            "playlist_id",
            "playlist_name",
            "playlist_acceptance_rate",
            "playlist_accepted_track_count",
        )
        if c in playlists_df.columns
    ]
    out = playlists_df[cols].copy() if cols else playlists_df.copy()
    if limit is not None:
        out = out.head(limit)
    return out
