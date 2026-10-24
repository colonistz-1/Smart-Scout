"""Model building, saving, and loading.

Keeping model construction in one place means train.py, predict.py, and the
Streamlit app can never disagree about how a model is built or loaded.
"""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Registry of supported model types. Add your own here and select them
# from config/config.yaml without touching the training script.
MODEL_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
}


def build_model(model_type: str, params: dict):
    """Instantiate a model from the registry with params from the config."""
    if model_type not in MODEL_REGISTRY:
        supported = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"Unknown model type '{model_type}'. Supported: {supported}")
    return MODEL_REGISTRY[model_type](**params)


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
