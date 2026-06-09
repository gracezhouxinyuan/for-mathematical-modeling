from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset' / 'descriptive_stats_demo.csv'
FIG = ROOT / 'figure'


def main() -> None:
    df = pd.read_csv(DATA)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    cols = df.columns.tolist()
    for ax, col in zip(axes.flat[:3], cols):
        ax.hist(df[col].dropna(), bins=18, color='#4C72B0', alpha=0.85)
        ax.set_title(f'{col} Histogram')
    df.boxplot(ax=axes.flat[3])
    axes.flat[3].set_title('Boxplot')
    plt.tight_layout()
    plt.savefig(FIG / 'fig_01_descriptive_statistics.png', dpi=220)
    plt.close()


if __name__ == '__main__':
    main()
