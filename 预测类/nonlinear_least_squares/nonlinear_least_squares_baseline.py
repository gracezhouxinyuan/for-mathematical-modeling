from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset' / 'nls_demo_data.csv'
OUT = ROOT / 'dataset' / 'results_baseline_fit.csv'


def model(x, a, b, c):
    return a * np.exp(b * x) + c


def main() -> None:
    df = pd.read_csv(DATA)
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    params, _ = curve_fit(model, x, y, p0=(2.0, -0.2, 0.5), maxfev=10000)
    y_hat = model(x, *params)
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    out = pd.DataFrame([{
        'method': 'baseline',
        'a': params[0], 'b': params[1], 'c': params[2],
        'rmse': rmse,
    }])
    out.to_csv(OUT, index=False)
    print(out)


if __name__ == '__main__':
    main()
