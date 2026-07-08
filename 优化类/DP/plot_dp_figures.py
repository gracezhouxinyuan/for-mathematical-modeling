from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figure"
DATA_DIR = ROOT / "dataset"

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_knapsack() -> None:
    base = pd.read_csv(DATA_DIR / "results_baseline_dp.csv")
    comp = pd.read_csv(DATA_DIR / "results_state_compress_dp.csv")

    fig, ax = plt.subplots(figsize=(8.2, 5))
    ax.plot(base["capacity"], base["best_value"], label="2D DP", linewidth=2)
    ax.plot(comp["capacity"], comp["best_value"], label="State Compressed DP", linewidth=2)
    ax.set_title("背包 DP 对比")
    ax.set_xlabel("Capacity")
    ax.set_ylabel("Best Value")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_01_dp_knapsack_comparison.png", dpi=180)
    plt.close(fig)


def plot_tsp_note() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.text(
        0.5,
        0.5,
        "Bitmask DP 适合小规模组合问题\n状态数约为 O(n·2^n)",
        ha="center",
        va="center",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.4", fc="#f8f8f8", ec="#444"),
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_02_dp_bitmask_note.png", dpi=180)
    plt.close(fig)


def main() -> None:
    plot_knapsack()
    plot_tsp_note()
    print("DP figures generated.")


if __name__ == "__main__":
    main()
