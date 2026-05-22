from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"


def normalize_matrix(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=0)
    return x / (denom + 1e-12)


def entropy_weights(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-12, None)
    p = x / x.sum(axis=0, keepdims=True)
    m = x.shape[0]
    k = 1.0 / np.log(m)
    e = -k * (p * np.log(p)).sum(axis=0)
    d = 1 - e
    return d / d.sum()


def grey_relational_grade(x: np.ndarray, ref: np.ndarray, rho: float = 0.5) -> np.ndarray:
    delta = np.abs(x - ref)
    delta_min = delta.min()
    delta_max = delta.max()
    grade = (delta_min + rho * delta_max) / (delta + rho * delta_max + 1e-12)
    return grade.mean(axis=1)


def topsis(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weighted = x * weights
    ideal_best = weighted.max(axis=0)
    ideal_worst = weighted.min(axis=0)
    d_plus = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    d_minus = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    return d_minus / (d_plus + d_minus + 1e-12)


def main() -> None:
    df = pd.read_csv(DATA_DIR / "topsis_demo_data.csv")
    x = df.drop(columns=["方案"]).astype(float).to_numpy()
    names = df["方案"]

    x[:, 0] = x[:, 0] / x[:, 0].max()
    x[:, 1] = x[:, 1] / x[:, 1].max()
    x[:, 2] = x[:, 2] / x[:, 2].max()
    x[:, 3] = x[:, 3] / x[:, 3].max()
    z = normalize_matrix(x)

    weights = entropy_weights(z)
    baseline_score = topsis(z, weights)

    ref = z.max(axis=0)
    gra = grey_relational_grade(z, ref).reshape(-1, 1)
    enhanced = z * 0.7 + gra * 0.3
    enhanced_score = topsis(enhanced, weights)

    result = pd.DataFrame(
        {
            "方案": names,
            "baseline": baseline_score,
            "gra_topsis": enhanced_score,
        }
    )
    result = result.sort_values("gra_topsis", ascending=False)
    result["排名"] = np.arange(1, len(result) + 1)
    result.to_csv(DATA_DIR / "results_gra_topsis.csv", index=False, encoding="utf-8-sig")

    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
