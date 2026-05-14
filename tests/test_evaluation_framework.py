"""Tests for the improved evaluation framework: NDCG@K, calibration (ECE),
slice analysis, and enhanced genre precision metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from matcher_agent.training.metrics import (
    CalibrationResult,
    RankingMetrics,
    SliceResult,
    _dcg,
    compute_calibration,
    grouped_ranking_metrics,
    median_split_column,
    slice_eval,
)


# ---------------------------------------------------------------------------
# NDCG@K tests
# ---------------------------------------------------------------------------


def test_dcg_perfect_ranking() -> None:
    """All relevant items at the top → DCG equals the maximum possible."""
    labels = np.array([1, 1, 0, 0, 0])
    dcg = _dcg(labels, k=5)
    assert dcg > 0.0
    # Ideal DCG for 2 relevant items at positions 1,2 should equal actual DCG.
    ideal = _dcg(np.array([1, 1, 0, 0, 0]), k=5)
    assert abs(dcg - ideal) < 1e-9


def test_dcg_worst_ranking() -> None:
    """Relevant items at the bottom → DCG is lower than best case."""
    best = _dcg(np.array([1, 1, 0, 0, 0]), k=5)
    worst = _dcg(np.array([0, 0, 0, 1, 1]), k=5)
    assert worst < best


def test_dcg_empty() -> None:
    """No relevant items → DCG is 0."""
    assert _dcg(np.array([0, 0, 0]), k=3) == 0.0


def test_ndcg_included_in_ranking_metrics() -> None:
    """grouped_ranking_metrics must include ndcg_at_k in the output."""
    eval_df = pd.DataFrame([
        {"track_id": "t1", "playlist_id": "p1", "label": 1, "pred_proba": 0.9},
        {"track_id": "t1", "playlist_id": "p2", "label": 0, "pred_proba": 0.1},
        {"track_id": "t1", "playlist_id": "p3", "label": 0, "pred_proba": 0.5},
    ])
    rm = grouped_ranking_metrics(eval_df, ks=(1, 3))
    assert "ndcg_at_1" in rm.as_flat_dict()
    assert "ndcg_at_3" in rm.as_flat_dict()
    assert rm.ndcg_at_k[1] > 0.0
    assert rm.ndcg_at_k[3] > 0.0


def test_ndcg_perfect_ranking_is_one() -> None:
    """When the only relevant item is ranked #1, NDCG@K should be 1.0."""
    eval_df = pd.DataFrame([
        {"track_id": "t1", "playlist_id": "p1", "label": 1, "pred_proba": 0.9},
        {"track_id": "t1", "playlist_id": "p2", "label": 0, "pred_proba": 0.1},
    ])
    rm = grouped_ranking_metrics(eval_df, ks=(1, 3))
    assert abs(rm.ndcg_at_k[1] - 1.0) < 1e-9
    assert abs(rm.ndcg_at_k[3] - 1.0) < 1e-9


def test_ndcg_imperfect_ranking() -> None:
    """When the relevant item is NOT at rank 1, NDCG < 1.0."""
    eval_df = pd.DataFrame([
        {"track_id": "t1", "playlist_id": "p1", "label": 0, "pred_proba": 0.9},
        {"track_id": "t1", "playlist_id": "p2", "label": 1, "pred_proba": 0.5},
        {"track_id": "t1", "playlist_id": "p3", "label": 0, "pred_proba": 0.1},
    ])
    rm = grouped_ranking_metrics(eval_df, ks=(3,))
    assert rm.ndcg_at_k[3] < 1.0
    assert rm.ndcg_at_k[3] > 0.0


def test_ranking_metrics_multiple_groups() -> None:
    """Metrics aggregate correctly over multiple track groups."""
    eval_df = pd.DataFrame([
        # Group 1: perfect ranking
        {"track_id": "t1", "playlist_id": "p1", "label": 1, "pred_proba": 0.9},
        {"track_id": "t1", "playlist_id": "p2", "label": 0, "pred_proba": 0.1},
        # Group 2: imperfect ranking
        {"track_id": "t2", "playlist_id": "p1", "label": 0, "pred_proba": 0.8},
        {"track_id": "t2", "playlist_id": "p2", "label": 1, "pred_proba": 0.3},
    ])
    rm = grouped_ranking_metrics(eval_df, ks=(1,))
    assert rm.n_groups == 2
    # t1 gets NDCG=1, t2 gets NDCG<1 → average is between 0.5 and 1.0.
    assert 0.4 < rm.ndcg_at_k[1] < 1.0
    # Hit@1: t1 hits, t2 misses → 0.5
    assert abs(rm.hit_at_k[1] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Calibration (ECE) tests
# ---------------------------------------------------------------------------


def test_calibration_perfect() -> None:
    """A perfectly calibrated model (pred == true freq) → ECE ≈ 0."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    cal = compute_calibration(y_true, y_prob, n_bins=2)
    assert cal.ece < 1e-9


def test_calibration_bad() -> None:
    """A poorly calibrated model (high confidence, all wrong) → high ECE."""
    y_true = np.zeros(100)
    y_prob = np.full(100, 0.9)
    cal = compute_calibration(y_true, y_prob, n_bins=10)
    assert cal.ece > 0.5


def test_calibration_result_fields() -> None:
    """CalibrationResult must have the expected fields."""
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.3, 0.7])
    cal = compute_calibration(y_true, y_prob, n_bins=5)
    assert cal.n_bins == 5
    assert len(cal.bin_edges) == 6
    assert len(cal.bin_true_freq) == 5
    assert len(cal.bin_pred_mean) == 5
    assert len(cal.bin_counts) == 5
    assert sum(cal.bin_counts) == 4


def test_calibration_as_flat_dict() -> None:
    y_true = np.array([0, 1])
    y_prob = np.array([0.3, 0.7])
    cal = compute_calibration(y_true, y_prob, n_bins=2)
    d = cal.as_flat_dict()
    assert "ece" in d
    assert "calibration_n_bins" in d


def test_calibration_empty_bins_handled() -> None:
    """If a bin has no samples, it should contribute 0 to ECE."""
    y_true = np.array([1, 1])
    y_prob = np.array([0.95, 0.99])
    cal = compute_calibration(y_true, y_prob, n_bins=10)
    assert cal.ece >= 0.0
    assert sum(1 for c in cal.bin_counts if c == 0) >= 5


# ---------------------------------------------------------------------------
# Slice analysis tests
# ---------------------------------------------------------------------------


def _make_slice_eval_df() -> pd.DataFrame:
    """Evaluation DataFrame with a slicing column."""
    rows = []
    for tid in ("t1", "t2", "t3", "t4"):
        for pid in ("p1", "p2", "p3"):
            label = 1 if pid == "p1" else 0
            score = 0.8 if pid == "p1" else 0.2
            rows.append({
                "track_id": tid, "playlist_id": pid,
                "label": label, "pred_proba": score,
                "category": "A" if tid in ("t1", "t2") else "B",
            })
    return pd.DataFrame(rows)


def test_slice_eval_returns_per_value_results() -> None:
    df = _make_slice_eval_df()
    results = slice_eval(df, slice_col="category", ks=(1, 3))
    assert len(results) == 2
    names = {r.slice_value for r in results}
    assert names == {"A", "B"}
    for r in results:
        assert r.n_groups > 0
        assert "mrr" in r.metrics


def test_slice_eval_handles_nan_values() -> None:
    df = _make_slice_eval_df()
    df.loc[0, "category"] = np.nan
    results = slice_eval(df, slice_col="category", ks=(1,))
    values = {r.slice_value for r in results}
    assert "unknown" in values


def test_median_split_column() -> None:
    df = pd.DataFrame({
        "track_id": ["t1", "t2", "t3", "t4"],
        "score": [10, 20, 30, 40],
    })
    result = median_split_column(df, "score", "bucket")
    assert set(result["bucket"]) == {"low", "high"}
    assert result.loc[result["score"] == 10, "bucket"].iloc[0] == "low"
    assert result.loc[result["score"] == 40, "bucket"].iloc[0] == "high"


def test_median_split_custom_labels() -> None:
    df = pd.DataFrame({"val": [1, 2, 3, 4]})
    result = median_split_column(df, "val", "bucket", labels=("small", "big"))
    assert "small" in result["bucket"].values
    assert "big" in result["bucket"].values


# ---------------------------------------------------------------------------
# RankingMetrics flat dict includes all fields
# ---------------------------------------------------------------------------


def test_ranking_metrics_flat_dict_complete() -> None:
    rm = RankingMetrics(
        n_groups=5, hit_at_k={1: 0.6}, precision_at_k={1: 0.6},
        recall_at_k={1: 0.6}, ndcg_at_k={1: 0.8}, mrr=0.7,
    )
    d = rm.as_flat_dict()
    assert d["mrr"] == 0.7
    assert d["ndcg_at_1"] == 0.8
    assert d["hit_at_1"] == 0.6
    assert d["precision_at_1"] == 0.6
    assert d["recall_at_1"] == 0.6
    assert d["n_groups"] == 5.0
