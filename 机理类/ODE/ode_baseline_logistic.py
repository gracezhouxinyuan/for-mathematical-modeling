from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "ode_logistic_observations.csv"
OUT = ROOT / "dataset" / "results_baseline_solution.csv"
METRICS = ROOT / "dataset" / "results_baseline_metrics.csv"


def logistic_rhs(_time, state, growth_rate=0.32, capacity=380.0):
    population = state[0]
    return [growth_rate * population * (1 - population / capacity)]


def solve_logistic(time_grid, initial_value):
    solution = solve_ivp(
        logistic_rhs,
        (time_grid.min(), time_grid.max()),
        [initial_value],
        t_eval=time_grid,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
    )
    return solution.y[0]


def main():
    observations = pd.read_csv(DATA)
    time_grid = observations["time"].to_numpy(dtype=float)
    predicted = solve_logistic(time_grid, observations["observed"].iloc[0])
    rmse = float(np.sqrt(mean_squared_error(observations["observed"], predicted)))

    result = pd.DataFrame({"time": time_grid, "observed": observations["observed"], "predicted": predicted})
    metrics = pd.DataFrame([{"model": "logistic_rk45", "rmse": rmse, "final_value": predicted[-1]}])
    result.to_csv(OUT, index=False)
    metrics.to_csv(METRICS, index=False)
    print(metrics)


if __name__ == "__main__":
    main()
