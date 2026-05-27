from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figure"
DATA_DIR = ROOT / "dataset"

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_convergence() -> None:
    base = pd.read_csv(DATA_DIR / "results_baseline_history.csv")
    imp = pd.read_csv(DATA_DIR / "results_improved_history.csv")

    fig, ax = plt.subplots(figsize=(8.2, 5))
    ax.plot(base["iteration"], base["best_value"], label="baseline PSO", linewidth=2)
    ax.plot(imp["iteration"], imp["best_value"], label="improved inertia-topology PSO", linewidth=2)
    ax.set_title("PSO 收敛曲线对比")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Value")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_01_pso_convergence.png", dpi=180)
    plt.close(fig)


def plot_metrics() -> None:
    base = pd.read_csv(DATA_DIR / "results_baseline_metrics.csv")
    imp = pd.read_csv(DATA_DIR / "results_improved_metrics.csv")
    df = pd.concat([base, imp], ignore_index=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(df["model"], df["best_value"], color=["#4c78a8", "#f58518"])
    ax.set_title("PSO 最优值对比")
    ax.set_ylabel("Best Value")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_02_pso_best_value.png", dpi=180)
    plt.close(fig)


def main() -> None:
    plot_convergence()
    plot_metrics()
    print("PSO figures generated.")


if __name__ == "__main__":
    main()
