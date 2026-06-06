from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figure"
DATA_DIR = ROOT / "dataset"

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_metrics() -> None:
    base = pd.read_csv(DATA_DIR / "results_baseline_metrics.csv")
    imp = pd.read_csv(DATA_DIR / "results_improved_metrics.csv")
    rad = pd.read_csv(DATA_DIR / "results_radius_metrics.csv")

    df = pd.DataFrame(
        {
            "model": ["baseline", "weighted", "radius"],
            "accuracy": [base.loc[0, "accuracy"], imp.loc[0, "accuracy"], rad.loc[0, "accuracy"]],
            "f1_macro": [base.loc[0, "f1_macro"], imp.loc[0, "f1_macro"], rad.loc[0, "f1_macro"]],
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(df["model"], df["accuracy"], color=["#4c78a8", "#f58518", "#54a24b"])
    axes[0].set_title("Accuracy 对比")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(df["model"], df["f1_macro"], color=["#4c78a8", "#f58518", "#54a24b"])
    axes[1].set_title("F1 对比")
    axes[1].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_01_knn_metric_comparison.png", dpi=180)
    plt.close(fig)


def plot_dataset() -> None:
    df = pd.read_csv(DATA_DIR / "knn_demo_data.csv")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    scatter = ax.scatter(df["f1"], df["f2"], c=df["label"], cmap="tab10", s=60)
    ax.set_title("KNN Demo Data")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_02_knn_dataset.png", dpi=180)
    plt.close(fig)


def main() -> None:
    plot_metrics()
    plot_dataset()
    print("KNN figures generated.")


if __name__ == "__main__":
    main()
