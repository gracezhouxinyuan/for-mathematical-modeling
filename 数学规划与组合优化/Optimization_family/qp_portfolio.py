from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"


def main() -> None:
    df = pd.read_csv(DATA_DIR / "qp_data.csv")
    mu = df["expected_return"].to_numpy(dtype=float)
    risk = df["risk"].to_numpy(dtype=float)

    def objective(w: np.ndarray) -> float:
        return float(np.sum(risk * w**2) - 0.5 * np.sum(mu * w))

    cons = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    )
    bounds = [(0.0, 1.0)] * len(mu)
    x0 = np.ones(len(mu)) / len(mu)

    res = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=cons)
    pd.DataFrame({"weight": res.x, "asset": df["asset"]}).to_csv(DATA_DIR / "results_qp_weights.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"success": [res.success], "objective": [res.fun]}).to_csv(DATA_DIR / "results_qp_metrics.csv", index=False, encoding="utf-8-sig")

    print(f"QP success={res.success}, objective={res.fun:.6f}")


if __name__ == "__main__":
    main()
