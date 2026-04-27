from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class RankingMetrics:
    n_groups: int
    hit_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    mrr: float

    def as_flat_dict(self) -> dict[str, float]:
        out: dict[str, float] = {"n_groups": float(self.n_groups), "mrr": self.mrr}
        for k, v in self.hit_at_k.items():
            out[f"hit_at_{k}"] = v
        for k, v in self.precision_at_k.items():
            out[f"precision_at_{k}"] = v
        for k, v in self.recall_at_k.items():
            out[f"recall_at_{k}"] = v
        return out


def grouped_ranking_metrics(
    eval_df: pd.DataFrame,
    *,
    group_col: str = "track_id",
    label_col: str = "label",
    score_col: str = "pred_proba",
    ks: Iterable[int] = (1, 3, 5, 10),
) -> RankingMetrics:
    """Compute per-group (per-track) ranking metrics over the test set.

    For each track we sort its candidate playlists by predicted probability
    descending, then compute hit/precision/recall@K and reciprocal rank of the
    first true-positive playlist.
    """
    ks = sorted(set(int(k) for k in ks))
    hit_acc: dict[int, list[float]] = defaultdict(list)
    prec_acc: dict[int, list[float]] = defaultdict(list)
    recall_acc: dict[int, list[float]] = defaultdict(list)
    rr_acc: list[float] = []

    for _, group in eval_df.groupby(group_col, sort=False):
        if group[label_col].sum() == 0:
            continue
        ranked = group.sort_values(score_col, ascending=False)
        labels = ranked[label_col].to_numpy()
        n_pos = int(labels.sum())
        first_hit = np.argmax(labels) if labels.any() else -1
        rr_acc.append(1.0 / (first_hit + 1) if labels.any() and labels[first_hit] == 1 else 0.0)
        for k in ks:
            top_k = labels[:k]
            hit_acc[k].append(1.0 if top_k.sum() > 0 else 0.0)
            prec_acc[k].append(top_k.sum() / k)
            recall_acc[k].append(top_k.sum() / n_pos if n_pos else 0.0)

    return RankingMetrics(
        n_groups=len(rr_acc),
        hit_at_k={k: float(np.mean(v)) if v else 0.0 for k, v in hit_acc.items()},
        precision_at_k={k: float(np.mean(v)) if v else 0.0 for k, v in prec_acc.items()},
        recall_at_k={k: float(np.mean(v)) if v else 0.0 for k, v in recall_acc.items()},
        mrr=float(np.mean(rr_acc)) if rr_acc else 0.0,
    )
