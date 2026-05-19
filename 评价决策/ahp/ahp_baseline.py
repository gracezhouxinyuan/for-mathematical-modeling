from __future__ import annotations

import numpy as np
import pandas as pd


RI_TABLE = {
    1: 0.0,
    2: 0.0,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}


def ahp_weights(matrix: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    eigvals, eigvecs = np.linalg.eig(matrix)
    max_idx = int(np.argmax(eigvals.real))
    lambda_max = float(eigvals[max_idx].real)
    weight = eigvecs[:, max_idx].real
    weight = np.abs(weight)
    weight = weight / weight.sum()

    n = matrix.shape[0]
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = RI_TABLE.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    return weight, lambda_max, ci, cr


def load_demo_data() -> tuple[np.ndarray, pd.DataFrame]:
    criteria_matrix = pd.read_csv("ahp/dataset/ahp_criteria_matrix.csv", index_col=0).values.astype(float)
    alternatives = pd.read_csv("ahp/dataset/ahp_alternatives_scores.csv")
    return criteria_matrix, alternatives


def score_alternatives(weights: np.ndarray, alt_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in alt_df.columns if c != "方案"]
    x = alt_df[feature_cols].values.astype(float)

    # 这里默认输入是“越大越好”指标，先做最小-最大归一化
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    x_norm = (x - x_min) / (x_max - x_min + 1e-12)

    scores = x_norm @ weights
    out = alt_df[["方案"]].copy()
    out["综合得分"] = scores
    out = out.sort_values("综合得分", ascending=False).reset_index(drop=True)
    out["排名"] = np.arange(1, len(out) + 1)
    return out


def main() -> None:
    matrix, alt_df = load_demo_data()
    weights, lambda_max, ci, cr = ahp_weights(matrix)

    print("=== Baseline AHP ===")
    print("权重:", np.round(weights, 4).tolist())
    print(f"lambda_max={lambda_max:.6f}, CI={ci:.6f}, CR={cr:.6f}")

    result = score_alternatives(weights, alt_df)
    print("\n方案排序:")
    print(result.to_string(index=False))

    pd.DataFrame(
        {
            "指标": ["经济效益", "可实施性", "可持续性"],
            "权重": weights,
        }
    ).to_csv("ahp/dataset/tab_02_ahp_weights_and_ranking.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        {
            "lambda_max": [lambda_max],
            "CI": [ci],
            "CR": [cr],
        }
    ).to_csv("ahp/dataset/tab_01_judgement_matrix_and_cr.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
