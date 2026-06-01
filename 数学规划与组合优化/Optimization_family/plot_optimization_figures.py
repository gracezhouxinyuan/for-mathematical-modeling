from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figure"
DATA_DIR = ROOT / "dataset"

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_lp_milp() -> None:
    lp = pd.read_csv(DATA_DIR / "results_lp.csv")
    milp = pd.read_csv(DATA_DIR / "results_milp_metrics.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(["LP"], lp["objective"], color="#4c78a8")
    axes[0].set_title("LP 目标值")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(["MILP"], milp["best_value"], color="#f58518")
    axes[1].set_title("MILP 目标值")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_01_lp_milp_compare.png", dpi=180)
    plt.close(fig)


def plot_qp_nlp() -> None:
    qp = pd.read_csv(DATA_DIR / "results_qp_metrics.csv")
    nlp = pd.read_csv(DATA_DIR / "results_nlp.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(["QP"], qp["objective"], color="#54a24b")
    axes[0].set_title("QP 目标值")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(["NLP"], nlp["objective"], color="#e45756")
    axes[1].set_title("NLP 目标值")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_02_qp_nlp_compare.png", dpi=180)
    plt.close(fig)


def main() -> None:
    plot_lp_milp()
    plot_qp_nlp()
    print("Optimization figures generated.")


if __name__ == "__main__":
    main()
