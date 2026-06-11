from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset'
FIG = ROOT / 'figure'


def main() -> None:
    base = pd.read_csv(DATA / 'results_baseline_state.csv')
    imp = pd.read_csv(DATA / 'results_improved_state.csv')
    true_pos = base['true_pos'].to_numpy()
    rmse_base = float(np.sqrt(np.mean((base['est_pos'] - true_pos) ** 2)))
    rmse_imp = float(np.sqrt(np.mean((imp['est_pos'] - true_pos) ** 2)))

    plt.figure(figsize=(10, 4))
    plt.plot(base['t'], true_pos, label='true', linewidth=2)
    plt.plot(base['t'], base['measurement'], label='measurement', alpha=0.4)
    plt.plot(base['t'], base['est_pos'], label=f'baseline RMSE={rmse_base:.2f}', linewidth=2)
    plt.plot(imp['t'], imp['est_pos'], label=f'improved RMSE={rmse_imp:.2f}', linewidth=2)
    plt.legend()
    plt.title('Kalman Filter Tracking')
    plt.tight_layout()
    plt.savefig(FIG / 'fig_01_kalman_tracking.png', dpi=220)
    plt.close()


if __name__ == '__main__':
    main()
