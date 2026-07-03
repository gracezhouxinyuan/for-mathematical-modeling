from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


@dataclass
class MetricRow:
    model: str
    mae: float
    rmse: float
    mape: float


def load_series(path: str) -> pd.Series:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    s = pd.Series(df["value"].values, index=df["date"])
    s = s.asfreq("MS")
    return s


def split_train_test(series: pd.Series, test_size: int = 12) -> tuple[pd.Series, pd.Series]:
    return series.iloc[:-test_size], series.iloc[-test_size:]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err / (y_true + 1e-12))) * 100)
    return mae, rmse, mape


def forecast_ar(train: pd.Series, steps: int) -> np.ndarray:
    model = AutoReg(train, lags=2, old_names=False).fit()
    pred = model.predict(start=len(train), end=len(train) + steps - 1)
    return np.asarray(pred)


def forecast_ma(train: pd.Series, steps: int) -> np.ndarray:
    model = ARIMA(train, order=(0, 0, 2)).fit()
    pred = model.forecast(steps=steps)
    return np.asarray(pred)


def forecast_arma(train: pd.Series, steps: int) -> np.ndarray:
    model = ARIMA(train, order=(2, 0, 1)).fit()
    pred = model.forecast(steps=steps)
    return np.asarray(pred)


def forecast_arima(train: pd.Series, steps: int) -> np.ndarray:
    model = ARIMA(train, order=(2, 1, 1)).fit()
    pred = model.forecast(steps=steps)
    return np.asarray(pred)


def forecast_sarima(train: pd.Series, steps: int) -> np.ndarray:
    model = ARIMA(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
    pred = model.forecast(steps=steps)
    return np.asarray(pred)


def main() -> None:
    series = load_series("arima_family/dataset/monthly_series_demo.csv")
    train, test = split_train_test(series, test_size=12)

    preds = {
        "AR": forecast_ar(train, len(test)),
        "MA": forecast_ma(train, len(test)),
        "ARMA": forecast_arma(train, len(test)),
        "ARIMA": forecast_arima(train, len(test)),
        "SARIMA": forecast_sarima(train, len(test)),
    }

    rows: list[MetricRow] = []
    for name, pred in preds.items():
        mae, rmse, mape = metrics(test.values, pred)
        rows.append(MetricRow(name, mae, rmse, mape))

    result = pd.DataFrame([r.__dict__ for r in rows]).sort_values("rmse")
    result.to_csv("arima_family/dataset/results_metrics.csv", index=False, encoding="utf-8-sig")

    pred_df = pd.DataFrame({"date": test.index, "actual": test.values})
    for name, pred in preds.items():
        pred_df[f"pred_{name}"] = pred
    pred_df.to_csv("arima_family/dataset/results_predictions.csv", index=False, encoding="utf-8-sig")

    print("模型误差对比：")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
