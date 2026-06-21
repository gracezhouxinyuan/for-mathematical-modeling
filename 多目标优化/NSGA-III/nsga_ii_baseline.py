from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'dataset' / 'results_baseline_nsga2.csv'

np.random.seed(42)


def objectives(x):
    f1 = x[0] ** 2 + x[1] ** 2
    f2 = (x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2
    return np.array([f1, f2])


def weighted_sum_ga(pop_size=60, generations=60, mutation_rate=0.15):
    pop = np.random.rand(pop_size, 2)
    weights = np.array([0.5, 0.5])
    best = None
    best_score = np.inf
    for _ in range(generations):
        scores = np.array([weights @ objectives(x) for x in pop])
        idx = np.argsort(scores)
        pop = pop[idx]
        if scores[idx[0]] < best_score:
            best_score = scores[idx[0]]
            best = pop[0].copy()
        elite = pop[:20]
        children = []
        while len(children) < pop_size:
            p1, p2 = elite[np.random.randint(len(elite))], elite[np.random.randint(len(elite))]
            alpha = np.random.rand()
            child = alpha * p1 + (1 - alpha) * p2
            if np.random.rand() < mutation_rate:
                child += np.random.normal(0, 0.08, size=2)
            child = np.clip(child, 0, 1)
            children.append(child)
        pop = np.array(children)
    f = objectives(best)
    return pd.DataFrame([{'method': 'baseline_weighted_ga', 'x1': best[0], 'x2': best[1], 'f1': f[0], 'f2': f[1], 'score': best_score}])


def main() -> None:
    out = weighted_sum_ga()
    out.to_csv(OUT, index=False)
    print(out)


if __name__ == '__main__':
    main()
