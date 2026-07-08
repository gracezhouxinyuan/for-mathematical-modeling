from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
FIG = ROOT / "figure"


def main():
    base = pd.read_csv(DATA / "results_baseline_predictions.csv")
    imp = pd.read_csv(DATA / "results_improved_predictions.csv")
    metrics = pd.concat([pd.read_csv(DATA / "results_baseline_metrics.csv"), pd.read_csv(DATA / "results_improved_metrics.csv")], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(base["y_true"].values, label="true", linewidth=2)
    axes[0].plot(base["y_pred"].values, label="baseline", alpha=0.8)
    axes[0].plot(imp["y_pred"].values, label="improved", alpha=0.8)
    axes[0].set_title("Forecast comparison")
    axes[0].legend()
    axes[1].bar(metrics["model"], metrics["mae"], color=["#4C72B0", "#55A868"])
    axes[1].set_title("MAE comparison")
    axes[1].tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(FIG / "fig_01_lstm_forecast.png", dpi=220)
    plt.close()


if __name__ == "__main__":
    main()
