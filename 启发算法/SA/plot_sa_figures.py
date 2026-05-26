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
    ax.plot(base["temperature"], base["best_length"], label="baseline SA", linewidth=2)
    ax.plot(imp["temperature"], imp["best_length"], label="improved SA", linewidth=2)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Best Route Length")
    ax.set_title("SA 收敛曲线对比")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_01_sa_convergence.png", dpi=180)
    plt.close(fig)


def plot_routes() -> None:
    cities = pd.read_csv(DATA_DIR / "sa_tsp_cities.csv")
    base = pd.read_csv(DATA_DIR / "results_baseline_route.csv")
    imp = pd.read_csv(DATA_DIR / "results_improved_route.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, route, title in [
        (axes[0], base, "baseline SA route"),
        (axes[1], imp, "improved SA route"),
    ]:
        ax.scatter(cities["x"], cities["y"], c="#4c78a8", s=55)
        route_x = route["x"].tolist() + [route["x"].iloc[0]]
        route_y = route["y"].tolist() + [route["y"].iloc[0]]
        ax.plot(route_x, route_y, "-o", color="#f58518")
        ax.set_title(title)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_02_sa_routes.png", dpi=180)
    plt.close(fig)


def main() -> None:
    plot_convergence()
    plot_routes()
    print("SA figures generated.")


if __name__ == "__main__":
    main()
