from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"

CAPACITY = 26


def main() -> None:
    items = pd.read_csv(DATA_DIR / "dp_knapsack_items.csv")
    weights = items["weight"].to_numpy(dtype=int)
    values = items["value"].to_numpy(dtype=int)
    n = len(items)

    dp = np.zeros((n + 1, CAPACITY + 1), dtype=int)

    for i in range(1, n + 1):
        w = weights[i - 1]
        v = values[i - 1]
        for c in range(CAPACITY + 1):
            dp[i, c] = dp[i - 1, c]
            if c >= w:
                dp[i, c] = max(dp[i, c], dp[i - 1, c - w] + v)

    result = pd.DataFrame(
        {
            "capacity": np.arange(CAPACITY + 1),
            "best_value": dp[n],
        }
    )
    result.to_csv(DATA_DIR / "results_baseline_dp.csv", index=False, encoding="utf-8-sig")
    print(f"baseline best value = {dp[n, CAPACITY]}")


if __name__ == "__main__":
    main()
