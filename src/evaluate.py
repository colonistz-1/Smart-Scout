"""Clustering evaluation and metric saving."""

import json
from pathlib import Path

from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
)


def compute_metrics(X_scaled, labels) -> dict:
    """Evaluate clustering in the standardized feature space."""
    return {
        "silhouette": float(silhouette_score(X_scaled, labels)),
        "davies_bouldin": float(
            davies_bouldin_score(X_scaled, labels)
        ),
    }


def save_metrics(metrics: dict, path) -> None:
    """Write metrics to a JSON file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )