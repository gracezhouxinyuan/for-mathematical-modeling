from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"

RANDOM_SEED = 42
N_SAMPLES = 8192


def estimate_pi_qmc(n: int) -> float:
    sampler = qmc.Sobol(d=2, scramble=True, seed=RANDOM_SEED)
    pts = sampler.random(n)
    inside = (pts[:, 0] ** 2 + pts[:, 1] ** 2) <= 1.0
    return float(4.0 * inside.mean())


def estimate_integral_qmc(n: int) -> float:
    sampler = qmc.Sobol(d=1, scramble=True, seed=RANDOM_SEED)
    x = sampler.random(n).ravel()
    fx = np.exp(-(x**2))
    return float(fx.mean())


def main() -> None:
    pi_est = estimate_pi_qmc(N_SAMPLES)
    integral_est = estimate_integral_qmc(N_SAMPLES)

    pd.DataFrame(
        {
            "method": ["qmc_pi", "qmc_integral"],
            "sample_size": [N_SAMPLES, N_SAMPLES],
            "estimate": [pi_est, integral_est],
        }
    ).to_csv(DATA_DIR / "results_qmc_metrics.csv", index=False, encoding="utf-8-sig")

    print(f"qmc pi estimate = {pi_est:.6f}")
    print(f"qmc integral estimate = {integral_est:.6f}")


if __name__ == "__main__":
    main()
