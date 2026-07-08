from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"

RANDOM_SEED = 42
PARTICLES = 30
ITERATIONS = 120
C1 = 2.05
C2 = 2.05
PHI = C1 + C2
CHI = 2 / abs(2 - PHI - np.sqrt(PHI**2 - 4 * PHI))
W_MAX = 0.9
W_MIN = 0.4


def rastrigin(x: np.ndarray) -> float:
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def load_benchmark() -> pd.Series:
    return pd.read_csv(DATA_DIR / "pso_benchmark.csv").iloc[0]


def local_best(pbest: np.ndarray, pbest_val: np.ndarray) -> np.ndarray:
    neighbors = []
    n = len(pbest)
    for i in range(n):
        idxs = [(i - 1) % n, i, (i + 1) % n]
        best_idx = idxs[int(np.argmin(pbest_val[idxs]))]
        neighbors.append(pbest[best_idx])
    return np.array(neighbors)


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    bench = load_benchmark()
    dim = int(bench["dimension"])
    lb = float(bench["lower_bound"])
    ub = float(bench["upper_bound"])

    x = np.random.uniform(lb, ub, size=(PARTICLES, dim))
    v = np.zeros((PARTICLES, dim))
    pbest = x.copy()
    pbest_val = np.array([rastrigin(i) for i in x])
    gbest_idx = int(np.argmin(pbest_val))
    gbest = pbest[gbest_idx].copy()
    gbest_val = float(pbest_val[gbest_idx])

    history = []
    path = []

    for t in range(ITERATIONS):
        w = W_MAX - (W_MAX - W_MIN) * (t / max(ITERATIONS - 1, 1))
        lbest = local_best(pbest, pbest_val)
        r1 = np.random.rand(PARTICLES, dim)
        r2 = np.random.rand(PARTICLES, dim)
        v = CHI * (w * v + C1 * r1 * (pbest - x) + C2 * r2 * (lbest - x))
        vmax = (ub - lb) * 0.2
        v = np.clip(v, -vmax, vmax)
        x = np.clip(x + v, lb, ub)

        vals = np.array([rastrigin(i) for i in x])
        better = vals < pbest_val
        pbest[better] = x[better]
        pbest_val[better] = vals[better]

        idx = int(np.argmin(pbest_val))
        if pbest_val[idx] < gbest_val:
            gbest_val = float(pbest_val[idx])
            gbest = pbest[idx].copy()

        history.append(gbest_val)
        path.append(gbest.copy())

    path_df = pd.DataFrame(path, columns=[f"x{i+1}" for i in range(dim)])
    path_df.insert(0, "iteration", np.arange(1, ITERATIONS + 1))
    path_df.to_csv(DATA_DIR / "results_improved_path.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        {
            "model": ["improved inertia-topology PSO"],
            "best_value": [gbest_val],
        }
    ).to_csv(DATA_DIR / "results_improved_metrics.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        {
            "iteration": np.arange(1, ITERATIONS + 1),
            "best_value": history,
        }
    ).to_csv(DATA_DIR / "results_improved_history.csv", index=False, encoding="utf-8-sig")

    print(f"improved best value = {gbest_val:.6f}, best position = {gbest.tolist()}")


if __name__ == "__main__":
    main()
