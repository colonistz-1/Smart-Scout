"""Train a model.

Entry point that wires together the reusable modules in src/ using the
settings in config/config.yaml. Keep this script thin — real logic belongs
in src/ where notebooks and the app can reuse it.

Usage:
    python scripts/train.py
    python scripts/train.py --config config/config_exp02.yaml
"""

import argparse
import sys
from pathlib import Path

# Allow "from src import ..." when running this file from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data, make_train_test_split, split_features_target
from src.evaluate import compute_metrics, save_confusion_matrix, save_metrics
from src.model import build_model, save_model
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    # 1. Data
    df = load_data(config["data"]["path"])
    X, y = split_features_target(df, config["data"]["target_column"])
    X_train, X_test, y_train, y_test = make_train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["training"]["random_state"],
    )
    print(f"Loaded {len(df)} rows -> {len(X_train)} train / {len(X_test)} test")

    # 2. Model
    model = build_model(config["model"]["type"], config["model"]["params"])
    model.fit(X_train, y_train)
    print(f"Trained {config['model']['type']} with {config['model']['params']}")

    # 3. Evaluation
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    save_metrics(metrics, config["output"]["metrics_path"])
    fig_path = save_confusion_matrix(y_test, y_pred, config["output"]["figures_dir"])
    print(f"Metrics: {metrics}  (saved to {config['output']['metrics_path']})")
    print(f"Confusion matrix saved to {fig_path}")

    # 4. Persist the model
    save_model(model, config["output"]["model_path"])
    print(f"Model saved to {config['output']['model_path']}")


if __name__ == "__main__":
    main()
