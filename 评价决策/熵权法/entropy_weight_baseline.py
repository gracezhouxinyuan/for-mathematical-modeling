from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset' / 'entropy_demo_data.csv'
OUT = ROOT / 'dataset' / 'results_baseline_scores.csv'


def minmax_normalize(df: pd.DataFrame, columns: list[str], cost_columns: set[str] | None = None) -> pd.DataFrame:
    cost_columns = cost_columns or set()
    normalized = df.copy()
    for col in columns:
        x = df[col].astype(float)
        if col in cost_columns:
            x = x.max() - x
        denom = x.max() - x.min()
        normalized[col] = 0.0 if denom == 0 else (x - x.min()) / denom
    return normalized


def compute_equal_weight_score(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    weights = np.full(len(columns), 1.0 / len(columns))
    scores = df[columns].to_numpy() @ weights
    result = df[['scheme']].copy()
    result['score'] = scores
    result['rank'] = result['score'].rank(ascending=False, method='min').astype(int)
    return result.sort_values('score', ascending=False).reset_index(drop=True)


def main() -> None:
    df = pd.read_csv(DATA)
    columns = ['cost', 'benefit1', 'benefit2', 'risk']
    normalized = minmax_normalize(df, columns, cost_columns={'cost', 'risk'})
    result = compute_equal_weight_score(normalized, columns)
    result.to_csv(OUT, index=False)
    print(result)


if __name__ == '__main__':
    main()
