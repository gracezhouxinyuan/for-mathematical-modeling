from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"

RANDOM_SEED = 42
N_SAMPLES = 10000


def stratified_pi(n: int) -> float:
    m = int(np.sqrt(n))
    if m * m != n:
        n = m * m
    grid = np.linspace(0, 1, m + 1)
    count = 0
    for i in range(m):
        for j in range(m):
            x = np.random.uniform(grid[i], grid[i + 1])
            y = np.random.uniform(grid[j], grid[j + 1])
            count += (x**2 + y**2) <= 1.0
    return float(4.0 * count / n)


def control_variate_integral(n: int) -> float:
    x = np.random.rand(n)
    fx = np.exp(-(x**2))
    gx = x
    eg = 0.5
    beta = np.cov(fx, gx, ddof=1)[0, 1] / (np.var(gx, ddof=1) + 1e-12)
    return float(np.mean(fx - beta * (gx - eg)))


def main() -> None:
    np.random.seed(RANDOM_SEED)

    pi_est = stratified_pi(N_SAMPLES)
    integral_est = control_variate_integral(N_SAMPLES)

    pd.DataFrame(
        {
            "method": ["stratified_pi", "control_variate_integral"],
            "sample_size": [N_SAMPLES, N_SAMPLES],
            "estimate": [pi_est, integral_est],
        }
    ).to_csv(DATA_DIR / "results_improved_metrics.csv", index=False, encoding="utf-8-sig")

    print(f"stratified pi estimate = {pi_est:.6f}")
    print(f"control variate integral estimate = {integral_est:.6f}")


if __name__ == "__main__":
    main()
