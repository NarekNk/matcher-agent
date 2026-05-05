from __future__ import annotations

from pathlib import Path

import pandas as pd

from matcher_agent.cli.plot_results import _metric_at_k_columns, _plot_ranking_at_k


def test_metric_at_k_columns_parses_row() -> None:
    row = pd.Series(
        {
            "n_groups": 10.0,
            "mrr": 0.5,
            "hit_at_1": 0.1,
            "hit_at_3": 0.2,
            "precision_at_1": 0.3,
            "precision_at_3": 0.4,
            "recall_at_1": 0.5,
            "recall_at_3": 0.6,
        }
    )
    assert _metric_at_k_columns(row, "hit") == {1: 0.1, 3: 0.2}
    assert _metric_at_k_columns(row, "precision") == {1: 0.3, 3: 0.4}
    assert _metric_at_k_columns(row, "recall") == {1: 0.5, 3: 0.6}


def test_plot_ranking_at_k_writes_png(tmp_path: Path) -> None:
    csv_path = tmp_path / "ranking_metrics.csv"
    pd.DataFrame(
        [
            {
                "hit_at_1": 0.5,
                "hit_at_3": 0.6,
                "precision_at_1": 0.4,
                "precision_at_3": 0.35,
                "recall_at_1": 0.2,
                "recall_at_3": 0.45,
            }
        ]
    ).to_csv(csv_path, index=False)
    out = _plot_ranking_at_k(csv_path, tmp_path)
    assert out is not None
    assert out.name == "ranking_hit_precision_recall_at_k.png"
    assert out.is_file()


def test_plot_ranking_at_k_returns_none_when_missing(tmp_path: Path) -> None:
    assert _plot_ranking_at_k(tmp_path / "missing.csv", tmp_path) is None
