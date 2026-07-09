from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
FIG = ROOT / "figure"


def main():
    transition = pd.read_csv(DATA / "results_improved_transition.csv", index_col=0)
    stationary = pd.read_csv(DATA / "results_stationary_distribution.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im = axes[0].imshow(transition.values, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(transition.columns)), transition.columns)
    axes[0].set_yticks(range(len(transition.index)), transition.index)
    axes[0].set_title("Transition matrix")
    for i in range(transition.shape[0]):
        for j in range(transition.shape[1]):
            axes[0].text(j, i, f"{transition.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    axes[1].bar(stationary["state"], stationary["stationary_probability"], color="#55A868")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Stationary distribution")
    plt.tight_layout()
    plt.savefig(FIG / "fig_01_markov_chain.png", dpi=220)
    plt.close()


if __name__ == "__main__":
    main()
