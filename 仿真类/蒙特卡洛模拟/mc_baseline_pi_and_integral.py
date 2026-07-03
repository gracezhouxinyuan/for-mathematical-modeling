from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"

RANDOM_SEED = 42
N_SAMPLES = 10000


def estimate_pi(n: int) -> float:
    x = np.random.rand(n)
    y = np.random.rand(n)
    inside = (x**2 + y**2) <= 1.0
    return float(4.0 * inside.mean())


def estimate_integral(n: int) -> float:
    x = np.random.rand(n)
    fx = np.exp(-(x**2))
    return float(fx.mean())


def main() -> None:
    np.random.seed(RANDOM_SEED)

    pi_est = estimate_pi(N_SAMPLES)
    integral_est = estimate_integral(N_SAMPLES)

    pd.DataFrame(
        {
            "task": ["pi_estimation", "integral_estimation"],
            "sample_size": [N_SAMPLES, N_SAMPLES],
            "estimate": [pi_est, integral_est],
        }
    ).to_csv(DATA_DIR / "results_baseline_metrics.csv", index=False, encoding="utf-8-sig")

    print(f"pi estimate = {pi_est:.6f}")
    print(f"integral estimate = {integral_est:.6f}")


if __name__ == "__main__":
    main()
