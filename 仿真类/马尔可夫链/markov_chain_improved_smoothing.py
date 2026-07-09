from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "markov_chain_demo_sequence.csv"
OUT = ROOT / "dataset" / "results_improved_transition.csv"
STAT = ROOT / "dataset" / "results_stationary_distribution.csv"


def estimate_transition(sequence, alpha=1.0):
    states = sorted(sequence.unique())
    counts = pd.DataFrame(alpha, index=states, columns=states)
    for a, b in zip(sequence[:-1], sequence[1:]):
        counts.loc[a, b] += 1
    return counts.div(counts.sum(axis=1), axis=0)


def stationary_distribution(P):
    n = P.shape[0]
    A = P.T - np.eye(n)
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    return np.linalg.solve(A, b)


def main():
    df = pd.read_csv(DATA)
    transition = estimate_transition(df["state"], alpha=1.0)
    pi = stationary_distribution(transition.to_numpy())
    transition.to_csv(OUT)
    pd.DataFrame({"state": transition.index, "stationary_probability": pi}).to_csv(STAT, index=False)
    print(transition)
    print(pd.DataFrame({"state": transition.index, "stationary_probability": pi}))


if __name__ == "__main__":
    main()
