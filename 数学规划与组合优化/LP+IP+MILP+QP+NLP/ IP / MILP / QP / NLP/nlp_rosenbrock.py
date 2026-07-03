from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"


def rosenbrock(x: np.ndarray) -> float:
    return float((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)


def main() -> None:
    x0 = np.array([-1.2, 1.0])
    res = minimize(rosenbrock, x0, method="BFGS")
    pd.DataFrame({"x1": [res.x[0]], "x2": [res.x[1]], "objective": [res.fun], "success": [res.success]}).to_csv(
        DATA_DIR / "results_nlp.csv", index=False, encoding="utf-8-sig"
    )
    print(f"NLP success={res.success}, objective={res.fun:.6f}")


if __name__ == "__main__":
    main()
