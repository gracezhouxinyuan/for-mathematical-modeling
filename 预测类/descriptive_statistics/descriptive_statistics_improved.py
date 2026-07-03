from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mstats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset' / 'descriptive_stats_demo.csv'
OUT = ROOT / 'dataset' / 'results_robust_summary.csv'


def winsorize_series(s: pd.Series, limits=(0.05, 0.05)) -> pd.Series:
    clean = s.copy()
    clean = clean.fillna(clean.median())
    return pd.Series(mstats.winsorize(clean, limits=limits), index=s.index)


def robust_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = winsorize_series(df[col])
        q1, q3 = s.quantile([0.25, 0.75])
        rows.append({
            'feature': col,
            'mean': s.mean(),
            'median': s.median(),
            'std': s.std(ddof=1),
            'mad': (s - s.median()).abs().median(),
            'iqr': q3 - q1,
            'missing_imputed': int(df[col].isna().sum()),
            'outlier_rate_iqr': float(((df[col] < (q1 - 1.5 * (q3 - q1))) | (df[col] > (q3 + 1.5 * (q3 - q1)))).mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(DATA)
    summary = robust_summary(df)
    summary.to_csv(OUT, index=False)
    print(summary)


if __name__ == '__main__':
    main()
