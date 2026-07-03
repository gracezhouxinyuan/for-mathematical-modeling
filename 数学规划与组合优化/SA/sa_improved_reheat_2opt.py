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
ALPHA = 0.97
ITER_PER_TEMP = 120
REHEAT_FACTOR = 0.6
STALL_LIMIT = 12


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


def two_opt(route: list[int], dist: np.ndarray) -> list[int]:
    best = route.copy()
    best_len = route_length(best, dist)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                cand = best[:i] + best[i:j][::-1] + best[j:]
                cand_len = route_length(cand, dist)
                if cand_len < best_len:
                    best = cand
                    best_len = cand_len
                    improved = True
    return best


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    cities = load_cities()
    coords = cities[["x", "y"]].to_numpy(dtype=float)
    dist = distance_matrix(coords)

    route = list(range(len(cities)))
    random.shuffle(route)
    route = two_opt(route, dist)
    current_length = route_length(route, dist)
    best_route = route.copy()
    best_length = current_length

    temp = INITIAL_TEMPERATURE
    history = []
    temp_history = []
    stall = 0

    while temp > FINAL_TEMPERATURE:
        improved_this_temp = False
        for _ in range(ITER_PER_TEMP):
            new_route = neighbor(route)
            new_route = two_opt(new_route, dist)
            new_length = route_length(new_route, dist)
            delta = new_length - current_length
            if delta < 0 or random.random() < math.exp(-delta / temp):
                route = new_route
                current_length = new_length
                improved_this_temp = True
                if current_length < best_length:
                    best_length = current_length
                    best_route = route.copy()
        history.append(best_length)
        temp_history.append(temp)

        if improved_this_temp:
            stall = 0
            temp *= ALPHA
        else:
            stall += 1
            temp = max(temp * REHEAT_FACTOR, FINAL_TEMPERATURE * 10)

        if stall >= STALL_LIMIT:
            temp = min(temp / REHEAT_FACTOR, INITIAL_TEMPERATURE)
            stall = 0

    pd.DataFrame(
        {
            "temperature": temp_history,
            "best_length": history,
        }
    ).to_csv(DATA_DIR / "results_improved_history.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        {
            "model": ["improved SA"],
            "best_length": [best_length],
        }
    ).to_csv(DATA_DIR / "results_improved_metrics.csv", index=False, encoding="utf-8-sig")

    route_df = pd.DataFrame(
        {
            "order": np.arange(1, len(best_route) + 1),
            "city": np.array(best_route) + 1,
            "x": coords[best_route, 0],
            "y": coords[best_route, 1],
        }
    )
    route_df.to_csv(DATA_DIR / "results_improved_route.csv", index=False, encoding="utf-8-sig")

    print(f"improved best length = {best_length:.6f}")


if __name__ == "__main__":
    main()
