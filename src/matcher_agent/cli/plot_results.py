from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _plot_calibration(calibration_csv: Path, output_dir: Path) -> Path:
    df = pd.read_csv(calibration_csv)
    out_path = output_dir / "calibration_curve.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["prob_pred"], df["prob_true"], marker="o", linewidth=2, label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Perfect calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_feature_importance(feature_importance_csv: Path, output_dir: Path, top_n: int) -> Path:
    df = pd.read_csv(feature_importance_csv)
    df = df.dropna(subset=["importance"]).sort_values("importance", ascending=False).head(top_n)
    out_path = output_dir / "feature_importance_top.png"
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(df))))
    ax.barh(df["feature"][::-1], df["importance"][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {len(df)} Feature Importances")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_lift_distribution(lift_csv: Path, output_dir: Path) -> Path:
    df = pd.read_csv(lift_csv)
    out_path = output_dir / "playlist_lift_distribution.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["lift"].dropna(), bins=40)
    ax.set_xlabel("Lift")
    ax.set_ylabel("Playlists")
    ax.set_title("Per-Playlist Lift Distribution")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_score_by_label(eval_csv: Path, output_dir: Path) -> Path:
    df = pd.read_csv(eval_csv)
    out_path = output_dir / "predicted_probability_by_label.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    neg = df[df["label"] == 0]["pred_proba"].dropna()
    pos = df[df["label"] == 1]["pred_proba"].dropna()
    ax.hist([neg, pos], bins=40, label=["declined (0)", "accepted (1)"], alpha=0.7)
    ax.set_xlabel("Predicted acceptance probability")
    ax.set_ylabel("Samples")
    ax.set_title("Predicted Probability by True Label")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training and evaluation diagnostics.")
    parser.add_argument("--output-dir", default="output/plots", help="Directory to save generated plots.")
    parser.add_argument("--eval-csv", default="output/ranker_eval.csv")
    parser.add_argument("--calibration-csv", default="output/calibration_curve.csv")
    parser.add_argument("--feature-importance-csv", default="output/feature_importance.csv")
    parser.add_argument("--lift-csv", default="output/per_playlist_lift.csv")
    parser.add_argument("--top-n-features", type=int, default=25)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_csv = Path(args.eval_csv)
    calibration_csv = Path(args.calibration_csv)
    feature_importance_csv = Path(args.feature_importance_csv)
    lift_csv = Path(args.lift_csv)

    print("[Plots] Generating calibration curve...")
    p1 = _plot_calibration(calibration_csv, output_dir)
    print("[Plots] Generating feature importance chart...")
    p2 = _plot_feature_importance(feature_importance_csv, output_dir, args.top_n_features)
    print("[Plots] Generating lift distribution...")
    p3 = _plot_lift_distribution(lift_csv, output_dir)
    print("[Plots] Generating score-by-label histogram...")
    p4 = _plot_score_by_label(eval_csv, output_dir)

    print("[Plots] Done.")
    print(
        {
            "calibration_curve": str(p1),
            "feature_importance": str(p2),
            "lift_distribution": str(p3),
            "score_by_label": str(p4),
        }
    )


if __name__ == "__main__":
    main()
