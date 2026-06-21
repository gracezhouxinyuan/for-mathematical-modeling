from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset'
FIG = ROOT / 'figure'


def main() -> None:
    base = pd.read_csv(DATA / 'results_baseline_nsga2.csv')
    imp = pd.read_csv(DATA / 'results_improved_nsga2.csv')
    plt.figure(figsize=(6, 5))
    plt.scatter(imp['f1'], imp['f2'], s=18, label='NSGA-II Pareto', color='#4C72B0')
    plt.scatter(base['f1'], base['f2'], s=80, label='Weighted GA', color='#C44E52', marker='x')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title('NSGA-II Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / 'fig_01_nsga_ii_pareto.png', dpi=220)
    plt.close()


if __name__ == '__main__':
    main()
