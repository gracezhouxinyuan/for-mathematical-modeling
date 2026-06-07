from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset'
FIG = ROOT / 'figure'


def main() -> None:
    baseline = pd.read_csv(DATA / 'results_baseline_predictions.csv')
    improved = pd.read_csv(DATA / 'results_improved_predictions.csv')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    RocCurveDisplay.from_predictions(baseline['y_true'], baseline['proba'], ax=axes[0], name='baseline')
    RocCurveDisplay.from_predictions(improved['y_true'], improved['proba'], ax=axes[0], name='improved')
    axes[0].set_title('ROC Curve')
    ConfusionMatrixDisplay.from_predictions(baseline['y_true'], baseline['pred'], ax=axes[1], cmap='Blues')
    axes[1].set_title('Baseline Confusion Matrix')
    plt.tight_layout()
    plt.savefig(FIG / 'fig_01_logistic_regression.png', dpi=220)
    plt.close()


if __name__ == '__main__':
    main()
