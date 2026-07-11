from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "ode_logistic_observations.csv"
OUT = ROOT / "dataset" / "results_solver_compare.csv"


def logistic_rhs(_time, state, growth_rate=0.32, capacity=380.0):
    population = state[0]
    return [growth_rate * population * (1 - population / capacity)]


def solve_with_method(method, time_grid, initial_value):
    start = time.perf_counter()
    solution = solve_ivp(
        logistic_rhs,
        (time_grid.min(), time_grid.max()),
        [initial_value],
        t_eval=time_grid,
        method=method,
        rtol=1e-7,
        atol=1e-9,
    )
    elapsed = time.perf_counter() - start
    return solution.y[0], elapsed, solution.nfev


def main():
    observations = pd.read_csv(DATA)
    time_grid = observations["time"].to_numpy(dtype=float)
    rows = []
    for method in ["RK23", "RK45", "DOP853", "Radau"]:
        predicted, elapsed, nfev = solve_with_method(method, time_grid, observations["observed"].iloc[0])
        rows.append({
            "method": method,
            "rmse": float(np.sqrt(mean_squared_error(observations["observed"], predicted))),
            "final_value": predicted[-1],
            "elapsed_seconds": elapsed,
            "nfev": nfev,
        })
    result = pd.DataFrame(rows).sort_values("rmse")
    result.to_csv(OUT, index=False)
    print(result)


if __name__ == "__main__":
    main()
