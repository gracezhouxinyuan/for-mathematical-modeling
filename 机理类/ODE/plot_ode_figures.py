from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
FIG = ROOT / "figure"


def main():
    solution = pd.read_csv(DATA / "results_baseline_solution.csv")
    compare = pd.read_csv(DATA / "results_solver_compare.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(solution["time"], solution["observed"], label="observed", color="#C44E52")
    axes[0].plot(solution["time"], solution["predicted"], label="RK45", color="#4C72B0")
    axes[0].set_title("Logistic ODE solution")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("state")
    axes[0].legend()

    axes[1].bar(compare["method"], compare["rmse"], color="#55A868")
    axes[1].set_title("Solver RMSE comparison")
    axes[1].set_ylabel("RMSE")
    plt.tight_layout()
    FIG.mkdir(exist_ok=True)
    plt.savefig(FIG / "fig_01_ode_logistic.png", dpi=220)
    plt.close()


if __name__ == "__main__":
    main()
