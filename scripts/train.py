"""Train baseline K-means clustering model

Usage:
    python scripts/train.py
    python scripts/train.py --config config/config_exp02.yaml
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_data
from src.preprocessing import preprocess_data, MODEL_FEATURES
from src.utils import load_config

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = ROOT / config_path

    # Load the configuration FIRST
    config = load_config(config_path)
    
    # 1. Load data
    data_path = ROOT / config["data"]["path"]
    df = load_data(data_path)

    # 2. Preprocess
    df = preprocess_data(
        df,
        min_minutes=config["data"]["min_minutes"]
    )

    X = df[MODEL_FEATURES]

    # 3. Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Train K-Means
    kmeans = KMeans(
        n_clusters=config["model"]["n_clusters"],
        random_state=config["training"]["random_state"],
        n_init=10
    )

    df["cluster"] = kmeans.fit_predict(X_scaled)

    # 5. Evaluate clusters
    print("\nCluster sizes:")
    print(df["cluster"].value_counts().sort_index())

    print("\nCluster profiles:")
    print(df.groupby("cluster")[MODEL_FEATURES].mean().round(3))

    if 1 < df["cluster"].nunique() < len(df):
        score = silhouette_score(X_scaled, df["cluster"])
        print(f"\nSilhouette score: {score:.3f}")
    else:
        print("\nSilhouette score unavailable")

    # 6. Save outputs
    output_path = ROOT / config["output"]["processed_path"]
    model_path = ROOT / config["output"]["model_path"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    joblib.dump(
        {
            "model": kmeans,
            "scaler": scaler,
            "features": MODEL_FEATURES,
        },
        model_path
    )

    print(f"\nSaved clustered data to {output_path}")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()