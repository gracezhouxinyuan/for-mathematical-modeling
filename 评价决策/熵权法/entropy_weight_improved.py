from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset' / 'entropy_demo_data.csv'
OUT = ROOT / 'dataset' / 'results_entropy_weight_scores.csv'
WEIGHT_OUT = ROOT / 'dataset' / 'results_entropy_weights.csv'


def minmax_normalize(df: pd.DataFrame, columns: list[str], cost_columns: set[str]) -> pd.DataFrame:
    normalized = df.copy()
    for col in columns:
        x = df[col].astype(float)
        if col in cost_columns:
            x = x.max() - x
        span = x.max() - x.min()
        normalized[col] = 0.0 if span == 0 else (x - x.min()) / span
    return normalized


def entropy_weights(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    x = df[columns].to_numpy(dtype=float)
    x = np.clip(x, 1e-12, None)
    p = x / x.sum(axis=0, keepdims=True)
    n = len(df)
    k = 1.0 / np.log(n)
    entropy = -k * np.sum(p * np.log(p), axis=0)
    divergence = 1 - entropy
    weights = divergence / divergence.sum()
    return pd.Series(weights, index=columns, name='weight')


def score(df: pd.DataFrame, columns: list[str], weights: pd.Series) -> pd.DataFrame:
    scores = df[columns].to_numpy() @ weights.to_numpy()
    result = df[['scheme']].copy()
    result['score'] = scores
    result['rank'] = result['score'].rank(ascending=False, method='min').astype(int)
    return result.sort_values('score', ascending=False).reset_index(drop=True)


def main() -> None:
    df = pd.read_csv(DATA)
    columns = ['cost', 'benefit1', 'benefit2', 'risk']
    normalized = minmax_normalize(df, columns, {'cost', 'risk'})
    weights = entropy_weights(normalized, columns)
    pd.DataFrame({'indicator': weights.index, 'weight': weights.to_numpy()}).to_csv(WEIGHT_OUT, index=False)
    result = score(normalized, columns, weights)
    result.to_csv(OUT, index=False)
    print(weights)
    print(result)


if __name__ == '__main__':
    main()
