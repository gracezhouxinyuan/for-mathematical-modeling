from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def compute_q_series(n_total: int, p: float, s_range: range) -> tuple[list[int], list[float], list[float], list[float]]:
    q_all: list[float] = []
    q1_all: list[float] = []
    q2_all: list[float] = []
    s_list = list(s_range)

    for s in s_list:
        q1 = np.floor(n_total / s)
        q2 = np.floor(n_total / s) * (1 - (1 - p) ** s) * s
        q = q1 + q2
        q1_all.append(float(q1))
        q2_all.append(float(q2))
        q_all.append(float(q))

    return s_list, q_all, q1_all, q2_all


def main() -> None:
    N = 25_000_000
    p = 0.01
    s_list, q, q1, q2 = compute_q_series(n_total=n_total, p=p, s_range=range(2, 30))

    plt.figure(figsize=(12, 5))
    plt.plot(s_list, q, "-o", label="Q")
    plt.plot(s_list, q1, "-o", label="Q1")
    plt.plot(s_list, q2, "-o", label="Q2")
    plt.hlines(y=n_total, xmin=2, xmax=30, linewidth=4, label="Num")
    plt.xlabel("s")
    plt.ylabel("Q")
    plt.xticks(range(2, 30))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
