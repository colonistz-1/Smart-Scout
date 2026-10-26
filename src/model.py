"""Model building, saving, and loading.

Keeping model construction in one place means train.py, predict.py, and the
Streamlit app can never disagree about how a model is built or loaded.
"""

from pathlib import Path

import joblib
from sklearn.cluster import KMeans


def build_model(model_type: str, params: dict):
    """Build a K-Means model."""
    if model_type != "kmeans":
        raise ValueError(
            f"Unknown model type '{model_type}'. Supported: kmeans"
        )
    return KMeans(**params)

def save_model(model, path: str) -> None:
    """Persist a trained model to disk (creates parent folders as needed)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    """Load a trained model saved by save_model()."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found at '{path}'. Run 'python scripts/train.py' first."
        )
    return joblib.load(model_path)
