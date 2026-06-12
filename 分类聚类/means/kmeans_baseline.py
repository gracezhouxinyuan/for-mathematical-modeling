from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"


def main() -> None:
    df = pd.read_csv(DATA_DIR / "kmeans_demo_data.csv")
    x = StandardScaler().fit_transform(df.to_numpy(dtype=float))

    model = KMeans(n_clusters=3, init="random", n_init=10, random_state=42)
    labels = model.fit_predict(x)

    result = pd.DataFrame(x, columns=df.columns)
    result["cluster"] = labels
    result.to_csv(DATA_DIR / "results_baseline_clusters.csv", index=False, encoding="utf-8-sig")

    print("inertia:", model.inertia_)
    print("silhouette:", silhouette_score(x, labels))
    print(result.head().to_string(index=False))


if __name__ == "__main__":
    main()
