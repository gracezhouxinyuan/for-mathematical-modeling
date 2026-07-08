from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def draw_hierarchy(path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axis("off")

    nodes = {
        "目标层\n最优方案选择": (0.5, 0.85),
        "准则1\n经济效益": (0.2, 0.55),
        "准则2\n可实施性": (0.5, 0.55),
        "准则3\n可持续性": (0.8, 0.55),
        "方案A": (0.15, 0.2),
        "方案B": (0.38, 0.2),
        "方案C": (0.62, 0.2),
        "方案D": (0.85, 0.2),
    }

    for label, (x, y) in nodes.items():
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7f7f7", "edgecolor": "#444"},
        )

    top = nodes["目标层\n最优方案选择"]
    for c in ["准则1\n经济效益", "准则2\n可实施性", "准则3\n可持续性"]:
        cx, cy = nodes[c]
        ax.plot([top[0], cx], [top[1] - 0.05, cy + 0.05], color="#444")

    for c in ["准则1\n经济效益", "准则2\n可实施性", "准则3\n可持续性"]:
        cx, cy = nodes[c]
        for alt in ["方案A", "方案B", "方案C", "方案D"]:
            axx, ayy = nodes[alt]
            ax.plot([cx, axx], [cy - 0.05, ayy + 0.05], color="#bbbbbb", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_matrix_heatmap(matrix: np.ndarray, labels: list[str], path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="YlOrRd")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black")

    ax.set_title("AHP 判断矩阵热力图")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_weight_comparison(labels: list[str], w1: np.ndarray, w2: np.ndarray, path: str) -> None:
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, w1, width, label="Baseline")
    ax.bar(x + width / 2, w2, width, label="Improved")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("权重")
    ax.set_title("基线与改良AHP权重对比")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_weights() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    matrix = pd.read_csv("ahp/dataset/ahp_criteria_matrix.csv", index_col=0).values.astype(float)
    labels = ["经济效益", "可实施性", "可持续性"]

    def eig_weight(m: np.ndarray) -> np.ndarray:
        eigvals, eigvecs = np.linalg.eig(m)
        idx = int(np.argmax(eigvals.real))
        w = np.abs(eigvecs[:, idx].real)
        return w / w.sum()

    baseline_w = eig_weight(matrix)

    fixed_path = "ahp/dataset/ahp_criteria_matrix_fixed.csv"
    try:
        fixed_matrix = pd.read_csv(fixed_path, index_col=0).values.astype(float)
    except FileNotFoundError:
        fixed_matrix = matrix
    improved_w = eig_weight(fixed_matrix)

    return matrix, baseline_w, improved_w, labels


def main() -> None:
    matrix, w1, w2, labels = load_weights()

    draw_hierarchy("ahp/figure/fig_01_ahp_hierarchy.png")
    draw_matrix_heatmap(matrix, labels, "ahp/figure/fig_02_judgement_matrix_heatmap.png")
    draw_weight_comparison(labels, w1, w2, "ahp/figure/fig_03_weight_comparison.png")

    print("图表生成完成：")
    print("- ahp/figure/fig_01_ahp_hierarchy.png")
    print("- ahp/figure/fig_02_judgement_matrix_heatmap.png")
    print("- ahp/figure/fig_03_weight_comparison.png")


if __name__ == "__main__":
    main()
