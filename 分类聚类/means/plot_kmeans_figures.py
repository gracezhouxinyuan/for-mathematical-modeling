from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figure"
DATA_DIR = ROOT / "dataset"

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_clusters() -> None:
    df = pd.read_csv(DATA_DIR / "kmeans_demo_data.csv")
    x = StandardScaler().fit_transform(df.to_numpy(dtype=float))
    model = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
    labels = model.fit_predict(x)
    centers = model.cluster_centers_

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    ax.scatter(x[:, 0], x[:, 1], c=labels, cmap="tab10", s=60)
    ax.scatter(centers[:, 0], centers[:, 1], c="black", marker="x", s=180, linewidths=3)
    ax.set_title("KMeans 聚类结果")
    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_01_kmeans_clusters.png", dpi=180)
    plt.close(fig)


def plot_metric_comparison() -> None:
    df = pd.read_csv(DATA_DIR / "results_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    axes[0].bar(df["model"], df["inertia"], color="#4c78a8")
    axes[0].set_title("Inertia 对比")
    axes[0].set_ylabel("inertia")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(df["model"], df["silhouette"], color="#54a24b")
    axes[1].set_title("Silhouette 对比")
    axes[1].set_ylabel("silhouette")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_02_kmeans_metric_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    plot_clusters()
    plot_metric_comparison()
    print("KMeans figures generated.")


if __name__ == "__main__":
    main()
