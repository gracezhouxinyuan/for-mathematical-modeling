from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"


def main() -> None:
    df = pd.read_csv(DATA_DIR / "gmm_demo_data.csv")
    x = StandardScaler().fit_transform(df.to_numpy(dtype=float))

    model = BayesianGaussianMixture(
        n_components=6,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        random_state=42,
        n_init=5,
        reg_covar=1e-6,
    )
    labels = model.fit_predict(x)
    probs = model.predict_proba(x)
    score = model.score(x)
    active_components = int(np.sum(model.weights_ > 0.05))

    out = pd.DataFrame(x, columns=["x1", "x2"])
    out["cluster"] = labels
    out.to_csv(DATA_DIR / "results_improved_clusters.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({"sample": np.arange(len(x)), "max_prob": probs.max(axis=1)}).to_csv(
        DATA_DIR / "results_improved_probabilities.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        {
            "model": ["BayesianGaussianMixture"],
            "avg_log_likelihood": [score],
            "active_components": [active_components],
        }
    ).to_csv(DATA_DIR / "results_improved_metrics.csv", index=False, encoding="utf-8-sig")

    print(f"improved avg log-likelihood = {score:.6f}, active components = {active_components}")


if __name__ == "__main__":
    main()
