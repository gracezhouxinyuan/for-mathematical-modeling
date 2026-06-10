from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset' / 'nls_demo_data.csv'
OUT = ROOT / 'dataset' / 'results_improved_fit.csv'


def model(params, x):
    a, b, c = params
    return a * np.exp(b * x) + c


def residuals(params, x, y):
    return model(params, x) - y


def main() -> None:
    df = pd.read_csv(DATA)
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    candidates = [(1.5, -0.1, 0.2), (3.0, -0.5, 0.6), (5.0, -1.0, 1.0)]
    best = None
    for p0 in candidates:
        res = least_squares(residuals, p0, args=(x, y), bounds=([0.0, -5.0, -5.0], [10.0, 1.0, 5.0]), loss='soft_l1')
        if best is None or res.cost < best.cost:
            best = res
    params = best.x
    y_hat = model(params, x)
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    out = pd.DataFrame([{
        'method': 'improved',
        'a': params[0], 'b': params[1], 'c': params[2],
        'rmse': rmse,
        'cost': best.cost,
        'nfev': best.nfev,
    }])
    out.to_csv(OUT, index=False)
    print(out)


if __name__ == '__main__':
    main()
