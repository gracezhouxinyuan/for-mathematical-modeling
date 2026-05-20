from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


def load_series(path: str) -> pd.Series:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    s = pd.Series(df["value"].values, index=df["date"])
    return s.asfreq("MS")


def split_train_test(series: pd.Series, test_size: int = 12) -> tuple[pd.Series, pd.Series]:
    return series.iloc[:-test_size], series.iloc[-test_size:]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err / (y_true + 1e-12))) * 100)
    return mae, rmse, mape


def main() -> None:
    series = load_series("arima_family/dataset/monthly_series_demo.csv")
    train, test = split_train_test(series)

    # 改良：明确季节项并检查残差白噪声
    model = ARIMA(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
    pred = np.asarray(model.forecast(steps=len(test)))

    mae, rmse, mape = metrics(test.values, pred)

    residuals = model.resid
    lb = acorr_ljungbox(residuals, lags=[6, 12], return_df=True)

    diagnostics = pd.DataFrame(
        {
            "metric": ["MAE", "RMSE", "MAPE", "LB_pvalue_lag6", "LB_pvalue_lag12"],
            "value": [mae, rmse, mape, float(lb.loc[6, "lb_pvalue"]), float(lb.loc[12, "lb_pvalue"])],
        }
    )
    diagnostics.to_csv("arima_family/dataset/results_sarima_diagnostics.csv", index=False, encoding="utf-8-sig")

    out_pred = pd.DataFrame({"date": test.index, "actual": test.values, "pred_sarima": pred})
    out_pred.to_csv("arima_family/dataset/results_sarima_predictions.csv", index=False, encoding="utf-8-sig")

    print("SARIMA 诊断结果：")
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
