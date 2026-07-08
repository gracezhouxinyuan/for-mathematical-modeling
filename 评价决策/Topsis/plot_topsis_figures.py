from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figure"
DATA_DIR = ROOT / "dataset"

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axis("off")
    boxes = [
        (0.05, 0.4, "原始指标"),
        (0.25, 0.4, "正向化"),
        (0.45, 0.4, "标准化"),
        (0.65, 0.4, "赋权"),
        (0.85, 0.4, "排序"),
    ]
    for x, y, text in boxes:
        ax.text(x, y, text, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#444"))
    for i in range(len(boxes) - 1):
        ax.annotate("", xy=(boxes[i + 1][0] - 0.06, 0.4), xytext=(boxes[i][0] + 0.06, 0.4), arrowprops=dict(arrowstyle="->", lw=1.5))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_01_topsis_pipeline.png", dpi=180)
    plt.close(fig)


def plot_ranking_bar() -> None:
    df = pd.read_csv(DATA_DIR / "results_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["方案"], df["贴近度"], color="#4c78a8")
    ax.set_ylabel("贴近度")
    ax.set_title("TOPSIS 排名结果")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_02_topsis_ranking_bar.png", dpi=180)
    plt.close(fig)


def plot_weight_sensitivity() -> None:
    df = pd.read_csv(DATA_DIR / "results_entropy_weights.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["指标"], df["权重"], color="#f58518")
    ax.set_ylabel("权重")
    ax.set_title("熵权分布")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_03_weight_sensitivity.png", dpi=180)
    plt.close(fig)


def main() -> None:
    plot_pipeline()
    plot_ranking_bar()
    plot_weight_sensitivity()
    print("TOPSIS figures generated.")


if __name__ == "__main__":
    main()
