from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"

CAPACITY = 26
ISLANDS = 3
POP_SIZE = 24
GENERATIONS = 120
MIGRATION_INTERVAL = 10
MIGRATION_NUM = 2
RANDOM_SEED = 42


def load_items() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ga_knapsack_items.csv")


def repair(individual: np.ndarray, items: pd.DataFrame) -> np.ndarray:
    weights = items["weight"].to_numpy()
    values = items["value"].to_numpy()
    while int(np.sum(individual * weights)) > CAPACITY:
        chosen = np.where(individual == 1)[0]
        ratios = values[chosen] / weights[chosen]
        drop_idx = chosen[int(np.argmin(ratios))]
        individual[drop_idx] = 0
    return individual


def fitness(individual: np.ndarray, items: pd.DataFrame) -> tuple[int, int]:
    weights = items["weight"].to_numpy()
    values = items["value"].to_numpy()
    total_weight = int(np.sum(individual * weights))
    total_value = int(np.sum(individual * values))
    return total_value, total_weight


def adaptive_rates(individual_score: int, avg_score: float, max_score: int) -> tuple[float, float]:
    if max_score <= avg_score:
        return 0.85, 0.04
    relative = (max_score - individual_score) / (max_score - avg_score + 1e-12)
    pc = 0.9 - 0.25 * np.clip(relative, 0, 1)
    pm = 0.02 + 0.10 * np.clip(relative, 0, 1)
    return float(pc), float(pm)


def tournament_selection(population: np.ndarray, scores: np.ndarray) -> np.ndarray:
    idx = np.random.choice(len(population), 3, replace=False)
    best_idx = idx[np.argmax(scores[idx])]
    return population[best_idx].copy()


def crossover(p1: np.ndarray, p2: np.ndarray, pc: float) -> tuple[np.ndarray, np.ndarray]:
    if random.random() > pc:
        return p1.copy(), p2.copy()
    a, b = sorted(random.sample(range(1, len(p1)), 2))
    c1 = p1.copy()
    c2 = p2.copy()
    c1[a:b] = p2[a:b]
    c2[a:b] = p1[a:b]
    return c1, c2


def mutate(individual: np.ndarray, pm: float) -> np.ndarray:
    mask = np.random.rand(len(individual)) < pm
    individual[mask] = 1 - individual[mask]
    return individual


def evaluate_population(population: np.ndarray, items: pd.DataFrame) -> np.ndarray:
    scores = []
    for ind in population:
        score, _ = fitness(ind, items)
        scores.append(score)
    return np.array(scores, dtype=float)


def evolve_island(population: np.ndarray, items: pd.DataFrame) -> tuple[np.ndarray, int, int]:
    population = np.array([repair(ind.copy(), items) for ind in population])
    scores = evaluate_population(population, items)
    avg_score = float(np.mean(scores))
    max_score = int(np.max(scores))
    elite_indices = np.argsort(scores)[-2:]
    new_population = list(population[elite_indices].copy())

    while len(new_population) < len(population):
        p1 = tournament_selection(population, scores)
        p2 = tournament_selection(population, scores)
        score1 = int(fitness(p1, items)[0])
        pc1, pm1 = adaptive_rates(score1, avg_score, max_score)
        c1, c2 = crossover(p1, p2, pc1)
        c1 = repair(mutate(c1, pm1), items)
        c2 = repair(mutate(c2, pm1), items)
        new_population.append(c1)
        if len(new_population) < len(population):
            new_population.append(c2)

    new_population = np.array(new_population[: len(population)])
    new_population = np.array([repair(ind.copy(), items) for ind in new_population])
    return new_population, int(np.max(scores)), int(np.mean(scores))


def migrate(islands: list[np.ndarray], items: pd.DataFrame) -> list[np.ndarray]:
    ranked = []
    for pop in islands:
        scores = evaluate_population(pop, items)
        order = np.argsort(scores)[::-1]
        ranked.append(pop[order])

    for i in range(len(ranked)):
        donor = ranked[i]
        receiver = ranked[(i + 1) % len(ranked)]
        receiver[-MIGRATION_NUM:] = donor[:MIGRATION_NUM].copy()
        ranked[(i + 1) % len(ranked)] = receiver
    return ranked


def run_mpga() -> tuple[np.ndarray, list[int], list[int]]:
    items = load_items()
    n_genes = len(items)
    islands = [
        np.array([repair(ind.copy(), items) for ind in np.random.randint(0, 2, size=(POP_SIZE, n_genes))])
        for _ in range(ISLANDS)
    ]
    best_history: list[int] = []
    avg_history: list[int] = []
    global_best = islands[0][0].copy()
    global_best_score = -10**9

    for gen in range(GENERATIONS):
        new_islands = []
        island_max_scores = []
        island_avg_scores = []
        for pop in islands:
            evolved, max_score, avg_score = evolve_island(pop, items)
            new_islands.append(evolved)
            island_max_scores.append(max_score)
            island_avg_scores.append(avg_score)

            idx = np.argmax(evaluate_population(evolved, items))
            score = int(fitness(evolved[idx], items)[0])
            if score > global_best_score:
                global_best_score = score
                global_best = evolved[idx].copy()

        islands = new_islands
        if (gen + 1) % MIGRATION_INTERVAL == 0:
            islands = migrate(islands, items)

        best_history.append(global_best_score)
        avg_history.append(int(np.mean(island_avg_scores)))

    return global_best, best_history, avg_history


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    best_individual, best_history, avg_history = run_mpga()
    items = load_items()
    best_value, best_weight = fitness(best_individual, items)

    pd.DataFrame(
        {
            "generation": np.arange(1, GENERATIONS + 1),
            "best_value": best_history,
            "avg_value": avg_history,
        }
    ).to_csv(DATA_DIR / "results_improved_history.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        {
            "model": ["improved AGA/MPGA"],
            "best_value": [best_value],
            "best_weight": [best_weight],
        }
    ).to_csv(DATA_DIR / "results_improved_metrics.csv", index=False, encoding="utf-8-sig")

    print(f"improved best value = {best_value}, best weight = {best_weight}")


if __name__ == "__main__":
    main()
