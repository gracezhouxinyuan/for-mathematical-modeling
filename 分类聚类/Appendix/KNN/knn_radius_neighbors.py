from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"


def main() -> None:
    df = pd.read_csv(DATA_DIR / "knn_demo_data.csv")
    x = df[["f1", "f2"]].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42, stratify=y)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = RadiusNeighborsClassifier(radius=1.5, weights="distance", outlier_label=0)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    pd.DataFrame(
        {
            "accuracy": [accuracy_score(y_test, pred)],
            "f1_macro": [f1_score(y_test, pred, average="macro")],
        }
    ).to_csv(DATA_DIR / "results_radius_metrics.csv", index=False, encoding="utf-8-sig")

    print(f"radius accuracy={accuracy_score(y_test, pred):.4f}")


if __name__ == "__main__":
    main()
