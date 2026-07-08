from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"


def main() -> None:
    # maximize 3x + 2y
    c = np.array([-3.0, -2.0])
    a_ub = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    b_ub = np.array([4.0, 2.0, 3.0], dtype=float)
    bounds = [(0, None)] * 2

    res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    pd.DataFrame(
        {
            "status": [res.status],
            "success": [res.success],
            "objective": [-res.fun if res.success else np.nan],
        }
    ).to_csv(DATA_DIR / "results_lp.csv", index=False, encoding="utf-8-sig")

    print(f"LP success={res.success}, objective={-res.fun if res.success else 'N/A'}")


if __name__ == "__main__":
    main()
