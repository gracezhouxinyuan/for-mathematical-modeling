from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset' / 'nls_demo_data.csv'
FIG = ROOT / 'figure'


def model(x, a, b, c):
    return a * np.exp(b * x) + c


def main() -> None:
    df = pd.read_csv(DATA)
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    base = pd.read_csv(ROOT / 'dataset' / 'results_baseline_fit.csv').iloc[0]
    imp = pd.read_csv(ROOT / 'dataset' / 'results_improved_fit.csv').iloc[0]
    xs = np.linspace(x.min(), x.max(), 200)

    plt.figure(figsize=(7, 4))
    plt.scatter(x, y, s=18, label='data', alpha=0.8)
    plt.plot(xs, model(xs, base['a'], base['b'], base['c']), label='baseline', linewidth=2)
    plt.plot(xs, model(xs, imp['a'], imp['b'], imp['c']), label='improved', linewidth=2)
    plt.legend()
    plt.title('Nonlinear Least Squares Fit')
    plt.tight_layout()
    plt.savefig(FIG / 'fig_01_nls_fit.png', dpi=220)
    plt.close()


if __name__ == '__main__':
    main()
