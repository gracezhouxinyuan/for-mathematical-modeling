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

    dp = np.zeros(CAPACITY + 1, dtype=int)
    history = []

    for w, v in zip(weights, values):
        for c in range(CAPACITY, w - 1, -1):
            dp[c] = max(dp[c], dp[c - w] + v)
        history.append(dp.copy())

    pd.DataFrame({"capacity": np.arange(CAPACITY + 1), "best_value": dp}).to_csv(
        DATA_DIR / "results_state_compress_dp.csv", index=False, encoding="utf-8-sig"
    )

    print(f"state-compressed best value = {dp[CAPACITY]}")


if __name__ == "__main__":
    main()
