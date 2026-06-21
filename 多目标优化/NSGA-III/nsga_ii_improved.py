from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'dataset' / 'results_improved_nsga2.csv'

np.random.seed(42)


def objectives(x):
    f1 = x[0] ** 2 + x[1] ** 2
    f2 = (x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2
    return np.array([f1, f2])


def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def fast_non_dominated_sort(F):
    S = [[] for _ in range(len(F))]
    n = [0] * len(F)
    rank = [0] * len(F)
    fronts = [[]]
    for p in range(len(F)):
        for q in range(len(F)):
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1], rank


def crowding_distance(F, front):
    if not front:
        return {}
    distance = {i: 0.0 for i in front}
    for m in range(F.shape[1]):
        vals = sorted(front, key=lambda i: F[i, m])
        distance[vals[0]] = distance[vals[-1]] = float('inf')
        fmin, fmax = F[vals[0], m], F[vals[-1], m]
        if fmax - fmin == 0:
            continue
        for k in range(1, len(vals) - 1):
            distance[vals[k]] += (F[vals[k + 1], m] - F[vals[k - 1], m]) / (fmax - fmin)
    return distance


def tournament_select(pop, F):
    i, j = np.random.randint(len(pop), size=2)
    fronts, rank = fast_non_dominated_sort(F)
    cd = {}
    for front in fronts:
        cd.update(crowding_distance(F, front))
    if rank[i] < rank[j]:
        return pop[i]
    if rank[j] < rank[i]:
        return pop[j]
    return pop[i] if cd.get(i, 0) > cd.get(j, 0) else pop[j]


def evolve(pop_size=60, generations=60, mutation_rate=0.15):
    pop = np.random.rand(pop_size, 2)
    for _ in range(generations):
        F = np.array([objectives(x) for x in pop])
        offspring = []
        while len(offspring) < pop_size:
            p1 = tournament_select(pop, F)
            p2 = tournament_select(pop, F)
            alpha = np.random.rand()
            child = alpha * p1 + (1 - alpha) * p2
            if np.random.rand() < mutation_rate:
                child += np.random.normal(0, 0.06, size=2)
            offspring.append(np.clip(child, 0, 1))
        combined = np.vstack([pop, np.array(offspring)])
        F = np.array([objectives(x) for x in combined])
        fronts, _ = fast_non_dominated_sort(F)
        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend(combined[front])
            else:
                dist = crowding_distance(F, front)
                ordered = sorted(front, key=lambda i: dist[i], reverse=True)
                need = pop_size - len(new_pop)
                new_pop.extend(combined[ordered[:need]])
                break
        pop = np.array(new_pop)
    F = np.array([objectives(x) for x in pop])
    fronts, _ = fast_non_dominated_sort(F)
    nd = fronts[0]
    out = pd.DataFrame({'x1': pop[nd, 0], 'x2': pop[nd, 1], 'f1': F[nd, 0], 'f2': F[nd, 1]})
    out['method'] = 'nsga_ii'
    return out.sort_values(['f1', 'f2']).reset_index(drop=True)


def main() -> None:
    out = evolve()
    out.to_csv(OUT, index=False)
    print(out.head())


if __name__ == '__main__':
    main()
