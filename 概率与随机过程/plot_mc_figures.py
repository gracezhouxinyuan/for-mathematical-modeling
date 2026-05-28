from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figure"
DATA_DIR = ROOT / "dataset"

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_metric_comparison() -> None:
    base = pd.read_csv(DATA_DIR / "results_baseline_metrics.csv")
    imp = pd.read_csv(DATA_DIR / "results_improved_metrics.csv")
    qmc = pd.read_csv(DATA_DIR / "results_qmc_metrics.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    merged_pi = pd.DataFrame(
        {
            "method": ["baseline", "improved", "qmc"],
            "estimate": [
                float(base.loc[base["task"] == "pi_estimation", "estimate"].iloc[0]),
                float(imp.loc[imp["method"] == "stratified_pi", "estimate"].iloc[0]),
                float(qmc.loc[qmc["method"] == "qmc_pi", "estimate"].iloc[0]),
            ],
        }
    )
    axes[0].bar(merged_pi["method"], merged_pi["estimate"], color=["#4c78a8", "#f58518", "#54a24b"])
    axes[0].axhline(3.1415926535, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("π 估计对比")
    axes[0].set_ylabel("estimate")
    axes[0].grid(axis="y", alpha=0.2)

    merged_int = pd.DataFrame(
        {
            "method": ["baseline", "improved", "qmc"],
            "estimate": [
                float(base.loc[base["task"] == "integral_estimation", "estimate"].iloc[0]),
                float(imp.loc[imp["method"] == "control_variate_integral", "estimate"].iloc[0]),
                float(qmc.loc[qmc["method"] == "qmc_integral", "estimate"].iloc[0]),
            ],
        }
    )
    axes[1].bar(merged_int["method"], merged_int["estimate"], color=["#4c78a8", "#f58518", "#54a24b"])
    axes[1].set_title("积分估计对比")
    axes[1].set_ylabel("estimate")
    axes[1].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_01_mc_estimation_comparison.png", dpi=180)
    plt.close(fig)


def plot_convergence_like() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.text(
        0.5,
        0.5,
        "Monte Carlo 收敛依赖样本量\n误差约随 1/sqrt(N) 下降",
        ha="center",
        va="center",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.4", fc="#f8f8f8", ec="#444"),
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_02_mc_convergence_note.png", dpi=180)
    plt.close(fig)


def main() -> None:
    plot_metric_comparison()
    plot_convergence_like()
    print("Monte Carlo figures generated.")


if __name__ == "__main__":
    main()
