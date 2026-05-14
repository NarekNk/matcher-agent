from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

@dataclass
class RankingMetrics:
    n_groups: int
    hit_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    mrr: float

    def as_flat_dict(self) -> dict[str, float]:
        out: dict[str, float] = {"n_groups": float(self.n_groups), "mrr": self.mrr}
        for k, v in self.hit_at_k.items():
            out[f"hit_at_{k}"] = v
        for k, v in self.precision_at_k.items():
            out[f"precision_at_{k}"] = v
        for k, v in self.recall_at_k.items():
            out[f"recall_at_{k}"] = v
        for k, v in self.ndcg_at_k.items():
            out[f"ndcg_at_{k}"] = v
        return out


def _dcg(relevances: np.ndarray, k: int) -> float:
    """Discounted Cumulative Gain for binary relevance labels."""
    top = relevances[:k].astype(np.float64)
    discounts = np.log2(np.arange(2, len(top) + 2))
    return float(np.sum(top / discounts))


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
    descending, then compute hit/precision/recall/NDCG@K and reciprocal rank
    of the first true-positive playlist.
    """
    ks = sorted(set(int(k) for k in ks))
    hit_acc: dict[int, list[float]] = defaultdict(list)
    prec_acc: dict[int, list[float]] = defaultdict(list)
    recall_acc: dict[int, list[float]] = defaultdict(list)
    ndcg_acc: dict[int, list[float]] = defaultdict(list)
    rr_acc: list[float] = []

    for _, group in eval_df.groupby(group_col, sort=False):
        if group[label_col].sum() == 0:
            continue
        ranked = group.sort_values(score_col, ascending=False)
        labels = ranked[label_col].to_numpy()
        n_pos = int(labels.sum())
        first_hit = np.argmax(labels) if labels.any() else -1
        rr_acc.append(1.0 / (first_hit + 1) if labels.any() and labels[first_hit] == 1 else 0.0)

        ideal_labels = np.sort(labels)[::-1]
        for k in ks:
            top_k = labels[:k]
            hit_acc[k].append(1.0 if top_k.sum() > 0 else 0.0)
            prec_acc[k].append(top_k.sum() / k)
            recall_acc[k].append(top_k.sum() / n_pos if n_pos else 0.0)

            dcg = _dcg(labels, k)
            idcg = _dcg(ideal_labels, k)
            ndcg_acc[k].append(dcg / idcg if idcg > 0 else 0.0)

    return RankingMetrics(
        n_groups=len(rr_acc),
        hit_at_k={k: float(np.mean(v)) if v else 0.0 for k, v in hit_acc.items()},
        precision_at_k={k: float(np.mean(v)) if v else 0.0 for k, v in prec_acc.items()},
        recall_at_k={k: float(np.mean(v)) if v else 0.0 for k, v in recall_acc.items()},
        ndcg_at_k={k: float(np.mean(v)) if v else 0.0 for k, v in ndcg_acc.items()},
        mrr=float(np.mean(rr_acc)) if rr_acc else 0.0,
    )


# ---------------------------------------------------------------------------
# Calibration analysis
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Expected Calibration Error (ECE) and per-bin reliability data."""

    ece: float
    n_bins: int
    bin_edges: list[float]
    bin_true_freq: list[float]
    bin_pred_mean: list[float]
    bin_counts: list[int]

    def as_flat_dict(self) -> dict[str, float]:
        return {"ece": self.ece, "calibration_n_bins": float(self.n_bins)}


def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> CalibrationResult:
    """Compute Expected Calibration Error (ECE) using uniform-width bins.

    ECE = sum_b ( |B_b| / N ) * |avg_confidence(B_b) - accuracy(B_b)|

    Also returns per-bin data suitable for plotting a reliability diagram.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_true_freq: list[float] = []
    bin_pred_mean: list[float] = []
    bin_counts: list[int] = []
    weighted_abs_diff = 0.0
    n_total = len(y_true)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if hi == bin_edges[-1]:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        count = int(mask.sum())
        bin_counts.append(count)
        if count == 0:
            bin_true_freq.append(0.0)
            bin_pred_mean.append(0.0)
            continue
        true_freq = float(y_true[mask].mean())
        pred_mean = float(y_prob[mask].mean())
        bin_true_freq.append(true_freq)
        bin_pred_mean.append(pred_mean)
        weighted_abs_diff += (count / n_total) * abs(true_freq - pred_mean)

    return CalibrationResult(
        ece=float(weighted_abs_diff),
        n_bins=n_bins,
        bin_edges=bin_edges.tolist(),
        bin_true_freq=bin_true_freq,
        bin_pred_mean=bin_pred_mean,
        bin_counts=bin_counts,
    )


# ---------------------------------------------------------------------------
# Slice analysis
# ---------------------------------------------------------------------------

@dataclass
class SliceResult:
    """Metrics for a single evaluation slice."""

    slice_name: str
    slice_value: str
    n_groups: int
    metrics: dict[str, float]


def slice_eval(
    eval_df: pd.DataFrame,
    *,
    slice_col: str,
    group_col: str = "track_id",
    label_col: str = "label",
    score_col: str = "pred_proba",
    ks: Iterable[int] = (1, 3, 5, 10),
) -> list[SliceResult]:
    """Split *eval_df* by *slice_col* values and compute ranking metrics
    for each bucket.

    ``slice_col`` must exist in *eval_df*. Rows where the column is NaN
    are grouped as ``"unknown"``.
    """
    df = eval_df.copy()
    df[slice_col] = df[slice_col].fillna("unknown").astype(str)
    results: list[SliceResult] = []
    for val, sub in df.groupby(slice_col, sort=True):
        rm = grouped_ranking_metrics(
            sub, group_col=group_col, label_col=label_col,
            score_col=score_col, ks=ks,
        )
        results.append(
            SliceResult(
                slice_name=slice_col,
                slice_value=str(val),
                n_groups=rm.n_groups,
                metrics=rm.as_flat_dict(),
            )
        )
    return results


def median_split_column(
    df: pd.DataFrame,
    src_col: str,
    dst_col: str,
    *,
    labels: tuple[str, str] = ("low", "high"),
) -> pd.DataFrame:
    """Add a binary column splitting *src_col* at the median."""
    median_val = df[src_col].median()
    df = df.copy()
    df[dst_col] = np.where(df[src_col] >= median_val, labels[1], labels[0])
    return df
