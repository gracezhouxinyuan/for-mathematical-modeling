from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"


def main() -> None:
    cities = np.array(
        [
            [0, 0],
            [2, 3],
            [5, 4],
            [6, 1],
            [8, 3],
        ],
        dtype=float,
    )
    n = len(cities)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(cities[i] - cities[j])

    size = 1 << n
    dp = np.full((size, n), math.inf)
    dp[1, 0] = 0.0

    for mask in range(size):
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            cur = dp[mask, last]
            if math.isinf(cur):
                continue
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                nmask = mask | (1 << nxt)
                dp[nmask, nxt] = min(dp[nmask, nxt], cur + dist[last, nxt])

    full = size - 1
    best = min(dp[full, j] + dist[j, 0] for j in range(n))
    pd.DataFrame({"best_tsp_length": [best]}).to_csv(DATA_DIR / "results_bitmask_tsp.csv", index=False, encoding="utf-8-sig")
    print(f"bitmask TSP best length = {best:.6f}")


if __name__ == "__main__":
    main()
