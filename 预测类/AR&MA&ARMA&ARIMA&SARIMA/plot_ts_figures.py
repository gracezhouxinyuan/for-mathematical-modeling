from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_train_test() -> None:
    df = pd.read_csv("arima_family/dataset/monthly_series_demo.csv")
    df["date"] = pd.to_datetime(df["date"])

    split_idx = len(df) - 12
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    plt.figure(figsize=(10, 4.8))
    plt.plot(train["date"], train["value"], label="Train")
    plt.plot(test["date"], test["value"], label="Test", linewidth=2)
    plt.title("时间序列训练/测试拆分")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("arima_family/figure/fig_01_series_train_test.png", dpi=180)
    plt.close()


def plot_forecast_comparison() -> None:
    df = pd.read_csv("arima_family/dataset/results_predictions.csv")
    df["date"] = pd.to_datetime(df["date"])

    plt.figure(figsize=(11, 5))
    plt.plot(df["date"], df["actual"], label="Actual", linewidth=2.5)
    for c in ["pred_AR", "pred_MA", "pred_ARMA", "pred_ARIMA", "pred_SARIMA"]:
        plt.plot(df["date"], df[c], label=c)

    plt.title("AR/MA/ARMA/ARIMA/SARIMA 预测对比")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend(ncol=3)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("arima_family/figure/fig_02_forecast_comparison.png", dpi=180)
    plt.close()


def plot_sarima_residual() -> None:
    df = pd.read_csv("arima_family/dataset/results_sarima_predictions.csv")
    residual = df["actual"] - df["pred_sarima"]

    plt.figure(figsize=(10, 4.8))
    plt.plot(residual.index, residual.values)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("SARIMA 残差序列")
    plt.xlabel("Index")
    plt.ylabel("Residual")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("arima_family/figure/fig_03_residual_diagnostics.png", dpi=180)
    plt.close()


def main() -> None:
    plot_train_test()
    plot_forecast_comparison()
    plot_sarima_residual()
    print("图表生成完成：")
    print("- arima_family/figure/fig_01_series_train_test.png")
    print("- arima_family/figure/fig_02_forecast_comparison.png")
    print("- arima_family/figure/fig_03_residual_diagnostics.png")


if __name__ == "__main__":
    main()
