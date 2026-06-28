from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import BisectingKMeans, KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"


def evaluate_model(name: str, model, x: np.ndarray) -> dict:
    labels = model.fit_predict(x)
    return {
        "model": name,
        "inertia": float(model.inertia_),
        "silhouette": float(silhouette_score(x, labels)),
    }


def main() -> None:
    df = pd.read_csv(DATA_DIR / "kmeans_demo_data.csv")
    x = StandardScaler().fit_transform(df.to_numpy(dtype=float))

    rows = [
        evaluate_model("KMeans++", KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42), x),
        evaluate_model("MiniBatchKMeans", MiniBatchKMeans(n_clusters=3, init="k-means++", random_state=42), x),
        evaluate_model("BisectingKMeans", BisectingKMeans(n_clusters=3, random_state=42), x),
    ]

    result = pd.DataFrame(rows).sort_values(["silhouette", "inertia"], ascending=[False, True])
    result.to_csv(DATA_DIR / "results_metrics.csv", index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
