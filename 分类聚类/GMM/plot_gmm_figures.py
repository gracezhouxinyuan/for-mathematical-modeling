from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figure"
DATA_DIR = ROOT / "dataset"

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_clusters() -> None:
    base = pd.read_csv(DATA_DIR / "results_baseline_clusters.csv")
    imp = pd.read_csv(DATA_DIR / "results_improved_clusters.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(base["x1"], base["x2"], c=base["cluster"], cmap="tab10", s=60)
    axes[0].set_title("Baseline GMM")
    axes[1].scatter(imp["x1"], imp["x2"], c=imp["cluster"], cmap="tab10", s=60)
    axes[1].set_title("Bayesian GMM")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_01_gmm_clusters.png", dpi=180)
    plt.close(fig)


def plot_metrics() -> None:
    base = pd.read_csv(DATA_DIR / "results_baseline_metrics.csv")
    imp = pd.read_csv(DATA_DIR / "results_improved_metrics.csv")
    df = pd.concat([base, imp], ignore_index=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(df["model"], df["avg_log_likelihood"], color=["#4c78a8", "#f58518"])
    ax.set_ylabel("avg log-likelihood")
    ax.set_title("GMM 与 Bayesian GMM 对比")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_02_gmm_metric_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    plot_clusters()
    plot_metrics()
    print("GMM figures generated.")


if __name__ == "__main__":
    main()
