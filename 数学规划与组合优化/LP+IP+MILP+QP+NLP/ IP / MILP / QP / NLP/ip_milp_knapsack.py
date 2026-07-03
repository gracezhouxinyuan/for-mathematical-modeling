from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"

CAPACITY = 15


def main() -> None:
    items = pd.read_csv(DATA_DIR / "milp_knapsack.csv")
    weights = items["weight"].to_numpy(dtype=float)
    values = items["value"].to_numpy(dtype=float)

    c = -values
    integrality = np.ones_like(c, dtype=int)
    constraints = LinearConstraint(weights.reshape(1, -1), -np.inf, CAPACITY)
    bounds = Bounds(np.zeros_like(c), np.ones_like(c))

    res = milp(c=c, integrality=integrality, constraints=constraints, bounds=bounds)
    x = np.round(res.x).astype(int)
    best_value = int(np.sum(x * values))
    best_weight = int(np.sum(x * weights))

    pd.DataFrame({"item": items["item"], "x": x}).to_csv(DATA_DIR / "results_milp_selection.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"best_value": [best_value], "best_weight": [best_weight]}).to_csv(DATA_DIR / "results_milp_metrics.csv", index=False, encoding="utf-8-sig")

    print(f"MILP best value={best_value}, best weight={best_weight}")


if __name__ == "__main__":
    main()
