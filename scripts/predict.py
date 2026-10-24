"""Predict with a trained model.

Loads the model saved by train.py and predicts on a CSV of new samples.
The CSV must contain the same feature columns that were used in training.

Usage:
    python scripts/predict.py --input data/new_samples.csv
    python scripts/predict.py --input data/new_samples.csv --output outputs/predictions.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow "from src import ..." when running this file from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import load_model
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV of samples to predict on")
    parser.add_argument("--output", default=None, help="Optional CSV path for results")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    model = load_model(config["output"]["model_path"])

    X_new = pd.read_csv(args.input)
    predictions = model.predict(X_new)

    result = X_new.copy()
    result["prediction"] = predictions
    print(result.to_string(index=False))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        print(f"\nPredictions saved to {args.output}")


if __name__ == "__main__":
    main()
