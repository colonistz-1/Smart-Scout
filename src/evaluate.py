"""Evaluation: metrics and figures.

Every run writes its metrics to outputs/ so experiments can be compared
after the fact instead of scrolling through terminal history.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render figures without a display (works on servers/CI)
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)


def compute_metrics(y_true, y_pred) -> dict:
    """Compute the headline metrics for a classification run."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
    }


def save_metrics(metrics: dict, path: str) -> None:
    """Write metrics as JSON to outputs/."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))


def save_confusion_matrix(y_true, y_pred, figures_dir: str) -> Path:
    """Plot and save a confusion matrix figure to outputs/figures/."""
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "confusion_matrix.png"

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path
