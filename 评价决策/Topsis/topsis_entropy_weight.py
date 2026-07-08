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
    x = x.copy()
    x = np.clip(x, 1e-12, None)
    p = x / x.sum(axis=0, keepdims=True)
    m = x.shape[0]
    k = 1.0 / np.log(m)
    e = -k * (p * np.log(p)).sum(axis=0)
    d = 1 - e
    return d / d.sum()


def topsis(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weighted = x * weights
    ideal_best = weighted.max(axis=0)
    ideal_worst = weighted.min(axis=0)
    d_plus = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    d_minus = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    return d_minus / (d_plus + d_minus + 1e-12)


def main() -> None:
    df = pd.read_csv(DATA_DIR / "topsis_demo_data.csv")
    names = df["方案"]
    x = df.drop(columns=["方案"]).astype(float).to_numpy()

    x[:, 0] = x[:, 0] / x[:, 0].max()
    x[:, 1] = x[:, 1] / x[:, 1].max()
    x[:, 2] = x[:, 2] / x[:, 2].max()
    x[:, 3] = x[:, 3] / x[:, 3].max()
    z = normalize_matrix(x)

    weights = entropy_weights(z)
    score = topsis(z, weights)

    result = pd.DataFrame({"方案": names, "熵权": score})
    result = result.sort_values("熵权", ascending=False)
    result["排名"] = np.arange(1, len(result) + 1)
    result.to_csv(DATA_DIR / "results_entropy_topsis.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({"指标": ["专著数量", "师资水平", "科研经费", "就业率"], "权重": weights}).to_csv(
        DATA_DIR / "results_entropy_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Entropy weights:", np.round(weights, 4).tolist())
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
