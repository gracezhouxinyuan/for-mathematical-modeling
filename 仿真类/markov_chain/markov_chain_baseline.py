from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "markov_chain_demo_sequence.csv"
OUT = ROOT / "dataset" / "results_baseline_transition.csv"
PRED = ROOT / "dataset" / "results_baseline_next_state.csv"


def estimate_transition(sequence):
    states = sorted(sequence.unique())
    counts = pd.DataFrame(0.0, index=states, columns=states)
    for a, b in zip(sequence[:-1], sequence[1:]):
        counts.loc[a, b] += 1
    transition = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    return transition


def main():
    df = pd.read_csv(DATA)
    transition = estimate_transition(df["state"])
    last_state = df["state"].iloc[-1]
    next_prob = transition.loc[last_state].rename("probability").reset_index().rename(columns={"index": "state"})
    transition.to_csv(OUT)
    next_prob.to_csv(PRED, index=False)
    print(transition)
    print(next_prob)


if __name__ == "__main__":
    main()
