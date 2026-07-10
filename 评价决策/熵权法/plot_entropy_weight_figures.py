from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figure'
DATA = ROOT / 'dataset'


def main() -> None:
    baseline = pd.read_csv(DATA / 'results_baseline_scores.csv')
    improved = pd.read_csv(DATA / 'results_entropy_weight_scores.csv')
    weights = pd.read_csv(DATA / 'results_entropy_weights.csv', header=None, names=['indicator', 'weight'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(weights['indicator'], weights['weight'], color='#4C72B0')
    axes[0].set_title('Entropy Weights')
    axes[0].set_ylabel('weight')
    axes[0].tick_params(axis='x', rotation=20)

    x = range(len(baseline))
    axes[1].bar([i - 0.18 for i in x], baseline['score'], width=0.36, label='baseline', color='#55A868')
    axes[1].bar([i + 0.18 for i in x], improved['score'], width=0.36, label='entropy', color='#C44E52')
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(baseline['scheme'])
    axes[1].set_title('Score Comparison')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIG / 'fig_01_entropy_weight_comparison.png', dpi=220)
    plt.close()


if __name__ == '__main__':
    main()
