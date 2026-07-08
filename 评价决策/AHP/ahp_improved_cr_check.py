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
    weight = np.abs(eigvecs[:, max_idx].real)
    weight = weight / weight.sum()

    n = matrix.shape[0]
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = RI_TABLE.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    return weight, lambda_max, ci, cr


def nearest_consistent_matrix(weights: np.ndarray) -> np.ndarray:
    """按 w_i / w_j 构造完全一致矩阵。"""
    n = len(weights)
    m = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            m[i, j] = weights[i] / weights[j]
    return m


def auto_fix_if_needed(matrix: np.ndarray, threshold: float = 0.1) -> tuple[np.ndarray, dict]:
    w, lam, ci, cr = ahp_weights(matrix)
    info = {
        "original_lambda_max": lam,
        "original_CI": ci,
        "original_CR": cr,
        "fixed": False,
    }

    if cr < threshold:
        return matrix, info

    fixed_matrix = nearest_consistent_matrix(w)
    w2, lam2, ci2, cr2 = ahp_weights(fixed_matrix)
    info.update(
        {
            "fixed": True,
            "fixed_lambda_max": lam2,
            "fixed_CI": ci2,
            "fixed_CR": cr2,
        }
    )
    return fixed_matrix, info


def score_alternatives(weights: np.ndarray, alt_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in alt_df.columns if c != "方案"]
    x = alt_df[feature_cols].values.astype(float)
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
    matrix_df = pd.read_csv("ahp/dataset/ahp_criteria_matrix.csv", index_col=0)
    matrix = matrix_df.values.astype(float)
    alt_df = pd.read_csv("ahp/dataset/ahp_alternatives_scores.csv")

    fixed_matrix, info = auto_fix_if_needed(matrix, threshold=0.1)
    weights, lam, ci, cr = ahp_weights(fixed_matrix)
    result = score_alternatives(weights, alt_df)

    print("=== Improved AHP (CR Check + Auto Fix) ===")
    print("是否触发修正:", info["fixed"])
    print(f"原始CR={info['original_CR']:.6f}")
    print(f"当前CR={cr:.6f}")
    print("权重:", np.round(weights, 4).tolist())
    print("\n方案排序:")
    print(result.to_string(index=False))

    matrix_out = pd.DataFrame(fixed_matrix, index=matrix_df.index, columns=matrix_df.columns)
    matrix_out.to_csv("ahp/dataset/ahp_criteria_matrix_fixed.csv", encoding="utf-8-sig")

    pd.DataFrame([info]).to_csv("ahp/dataset/tab_01_judgement_matrix_and_cr.csv", index=False, encoding="utf-8-sig")

    result.to_csv("ahp/dataset/tab_02_ahp_weights_and_ranking.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
