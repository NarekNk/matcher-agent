from __future__ import annotations

import pandas as pd

DEFAULT_STATUS_MAP = {
    "accepted": 1,
    "closed": 1,
    "approve": 1,
    "approved": 1,
    "declined": 0,
    "rejected": 0,
    "reject": 0,
}


def map_status_to_label(
    df: pd.DataFrame,
    *,
    status_col: str = "status",
    output_col: str = "label",
    mapping: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = mapping or DEFAULT_STATUS_MAP
    out = df.copy()
    statuses = out[status_col].fillna("").astype(str).str.strip().str.lower()
    out[output_col] = statuses.map(mapping)
    rejects = out[out[output_col].isna()].copy()
    accepted = out[out[output_col].notna()].copy()
    accepted[output_col] = accepted[output_col].astype(int)
    return accepted, rejects
