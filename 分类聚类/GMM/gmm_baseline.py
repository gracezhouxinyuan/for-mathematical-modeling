from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"


def main() -> None:
    df = pd.read_csv(DATA_DIR / "gmm_demo_data.csv")
    x = StandardScaler().fit_transform(df.to_numpy(dtype=float))

    model = GaussianMixture(n_components=3, covariance_type="full", random_state=42, n_init=5)
    labels = model.fit_predict(x)
    probs = model.predict_proba(x)
    score = model.score(x)

    out = pd.DataFrame(x, columns=["x1", "x2"])
    out["cluster"] = labels
    out.to_csv(DATA_DIR / "results_baseline_clusters.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({"sample": np.arange(len(x)), "max_prob": probs.max(axis=1)}).to_csv(
        DATA_DIR / "results_baseline_probabilities.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"model": ["GaussianMixture"], "avg_log_likelihood": [score]}).to_csv(
        DATA_DIR / "results_baseline_metrics.csv", index=False, encoding="utf-8-sig"
    )

    print(f"baseline avg log-likelihood = {score:.6f}")


if __name__ == "__main__":
    main()
