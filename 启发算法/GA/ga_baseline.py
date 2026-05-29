from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
FIG_DIR = ROOT / "figure"

CAPACITY = 26
POP_SIZE = 60
GENERATIONS = 120
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.03
TOURNAMENT_SIZE = 3
RANDOM_SEED = 42


def load_items() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ga_knapsack_items.csv")


def fitness(individual: np.ndarray, items: pd.DataFrame) -> tuple[int, int]:
    weights = items["weight"].to_numpy()
    values = items["value"].to_numpy()
    total_weight = int(np.sum(individual * weights))
    total_value = int(np.sum(individual * values))
    if total_weight > CAPACITY:
        penalty = (total_weight - CAPACITY) * 20
        return total_value - penalty, total_weight
    return total_value, total_weight


def tournament_selection(population: np.ndarray, scores: np.ndarray) -> np.ndarray:
    idx = np.random.choice(len(population), TOURNAMENT_SIZE, replace=False)
    best_idx = idx[np.argmax(scores[idx])]
    return population[best_idx].copy()


def crossover(p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() > CROSSOVER_RATE:
        return p1.copy(), p2.copy()
    point = random.randint(1, len(p1) - 1)
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1, c2


def mutate(individual: np.ndarray) -> np.ndarray:
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]
    return individual


def evaluate_population(population: np.ndarray, items: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    weights = []
    for ind in population:
        score, weight = fitness(ind, items)
        scores.append(score)
        weights.append(weight)
    return np.array(scores), np.array(weights)


def run_ga() -> tuple[np.ndarray, list[int], list[int], list[int]]:
    items = load_items()
    n_genes = len(items)
    population = np.random.randint(0, 2, size=(POP_SIZE, n_genes))
    best_history: list[int] = []
    weight_history: list[int] = []
    value_history: list[int] = []

    best_individual = population[0].copy()
    best_score = -10**9

    for _ in range(GENERATIONS):
        scores, weights = evaluate_population(population, items)
        idx = int(np.argmax(scores))
        if scores[idx] > best_score:
            best_score = int(scores[idx])
            best_individual = population[idx].copy()

        best_history.append(int(best_score))
        weight_history.append(int(fitness(best_individual, items)[1]))
        value_history.append(int(np.sum(best_individual * items["value"].to_numpy())))

        new_population = []
        elite_indices = np.argsort(scores)[-2:]
        new_population.extend(population[elite_indices].copy())

        while len(new_population) < POP_SIZE:
            p1 = tournament_selection(population, scores)
            p2 = tournament_selection(population, scores)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_population.append(c1)
            if len(new_population) < POP_SIZE:
                new_population.append(c2)

        population = np.array(new_population[:POP_SIZE])

    return best_individual, best_history, weight_history, value_history


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    best_individual, best_history, weight_history, value_history = run_ga()
    items = load_items()

    best_weight = int(np.sum(best_individual * items["weight"].to_numpy()))
    best_value = int(np.sum(best_individual * items["value"].to_numpy()))

    pd.DataFrame(
        {
            "generation": np.arange(1, GENERATIONS + 1),
            "best_value": best_history,
            "best_weight": weight_history,
            "best_individual_value": value_history,
        }
    ).to_csv(DATA_DIR / "results_baseline_history.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        {
            "model": ["baseline GA"],
            "best_value": [best_value],
            "best_weight": [best_weight],
        }
    ).to_csv(DATA_DIR / "results_baseline_metrics.csv", index=False, encoding="utf-8-sig")

    print(f"baseline best value = {best_value}, best weight = {best_weight}")


if __name__ == "__main__":
    main()
