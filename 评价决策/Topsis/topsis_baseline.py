from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"


def normalize_matrix(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=0)
    return x / (denom + 1e-12)


def topsis(x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weighted = x * weights
    ideal_best = weighted.max(axis=0)
    ideal_worst = weighted.min(axis=0)
    d_plus = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    d_minus = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    score = d_minus / (d_plus + d_minus + 1e-12)
    return score, d_plus, d_minus


def main() -> None:
    df = pd.read_csv(DATA_DIR / "topsis_demo_data.csv")
    names = df["方案"]
    x = df.drop(columns=["方案"]).copy()

    x["专著数量"] = x["专著数量"].astype(float)
    x["师资水平"] = x["师资水平"].astype(float)
    x["科研经费"] = x["科研经费"].astype(float)
    x["就业率"] = x["就业率"].astype(float)

    x_np = x.to_numpy(dtype=float)
    x_np[:, 0] = x_np[:, 0] / x_np[:, 0].max()
    x_np[:, 1] = x_np[:, 1] / x_np[:, 1].max()
    x_np[:, 2] = x_np[:, 2] / x_np[:, 2].max()
    x_np[:, 3] = x_np[:, 3] / x_np[:, 3].max()

    z = normalize_matrix(x_np)
    weights = np.array([0.2, 0.3, 0.3, 0.2], dtype=float)
    score, d_plus, d_minus = topsis(z, weights)

    result = pd.DataFrame(
        {
            "方案": names,
            "D_plus": d_plus,
            "D_minus": d_minus,
            "贴近度": score,
        }
    ).sort_values("贴近度", ascending=False)
    result["排名"] = np.arange(1, len(result) + 1)
    result.to_csv(DATA_DIR / "results_metrics.csv", index=False, encoding="utf-8-sig")

    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
