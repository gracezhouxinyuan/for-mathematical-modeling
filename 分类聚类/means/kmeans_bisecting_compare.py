from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"


def main() -> None:
    df = pd.read_csv(DATA_DIR / "kmeans_demo_data.csv")
    x = StandardScaler().fit_transform(df.to_numpy(dtype=float))

    km = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
    bk = BisectingKMeans(n_clusters=3, random_state=42)

    labels_km = km.fit_predict(x)
    labels_bk = bk.fit_predict(x)

    out = pd.DataFrame(
        [
            {"model": "KMeans++", "inertia": km.inertia_, "silhouette": silhouette_score(x, labels_km)},
            {"model": "BisectingKMeans", "inertia": bk.inertia_, "silhouette": silhouette_score(x, labels_bk)},
        ]
    )
    out.to_csv(DATA_DIR / "results_bisect_compare.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
