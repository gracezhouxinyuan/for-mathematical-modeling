from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset' / 'descriptive_stats_demo.csv'
OUT = ROOT / 'dataset' / 'results_baseline_summary.csv'


def main() -> None:
    df = pd.read_csv(DATA)
    summary = df.describe(percentiles=[0.25, 0.5, 0.75]).T
    summary['missing_count'] = df.isna().sum()
    summary['skew'] = df.skew(numeric_only=True)
    summary['kurtosis'] = df.kurtosis(numeric_only=True)
    summary.to_csv(OUT)
    print(summary)


if __name__ == '__main__':
    main()
