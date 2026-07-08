from __future__ import annotations

from pathlib import Path
import math
import random

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"

RANDOM_SEED = 42
INITIAL_TEMPERATURE = 200.0
FINAL_TEMPERATURE = 1e-3
ALPHA = 0.90
ITER_PER_TEMP = 60


def load_cities() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sa_tsp_cities.csv")


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    n = len(coords)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(coords[i] - coords[j])
    return dist


def route_length(route: list[int], dist: np.ndarray) -> float:
    total = 0.0
    for i in range(len(route)):
        a = route[i]
        b = route[(i + 1) % len(route)]
        total += dist[a, b]
    return total


def neighbor(route: list[int]) -> list[int]:
    i, j = sorted(random.sample(range(len(route)), 2))
    new_route = route.copy()
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    cities = load_cities()
    coords = cities[["x", "y"]].to_numpy(dtype=float)
    dist = distance_matrix(coords)

    route = list(range(len(cities)))
    random.shuffle(route)
    current_length = route_length(route, dist)
    best_route = route.copy()
    best_length = current_length

    temp = INITIAL_TEMPERATURE
    history = []
    temp_history = []

    while temp > FINAL_TEMPERATURE:
        for _ in range(ITER_PER_TEMP):
            new_route = neighbor(route)
            new_length = route_length(new_route, dist)
            delta = new_length - current_length
            if delta < 0 or random.random() < math.exp(-delta / temp):
                route = new_route
                current_length = new_length
                if current_length < best_length:
                    best_length = current_length
                    best_route = route.copy()
        history.append(best_length)
        temp_history.append(temp)
        temp *= ALPHA

    pd.DataFrame(
        {
            "temperature": temp_history,
            "best_length": history,
        }
    ).to_csv(DATA_DIR / "results_baseline_history.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        {
            "model": ["baseline SA"],
            "best_length": [best_length],
        }
    ).to_csv(DATA_DIR / "results_baseline_metrics.csv", index=False, encoding="utf-8-sig")

    route_df = pd.DataFrame(
        {
            "order": np.arange(1, len(best_route) + 1),
            "city": np.array(best_route) + 1,
            "x": coords[best_route, 0],
            "y": coords[best_route, 1],
        }
    )
    route_df.to_csv(DATA_DIR / "results_baseline_route.csv", index=False, encoding="utf-8-sig")

    print(f"baseline best length = {best_length:.6f}")


if __name__ == "__main__":
    main()
