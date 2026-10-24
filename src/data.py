"""Data loading and splitting.

Anything that reads, cleans, or splits data lives here so that scripts,
notebooks, and the app all load data the exact same way.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file.

    Falls back to the built-in Iris dataset when the file does not exist,
    so the template runs before you have added any data of your own.
    Replace this fallback with a hard error (or a download step) in a real
    project.
    """
    csv_path = Path(path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    from sklearn.datasets import load_iris

    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "species"})
    df["species"] = df["species"].map(dict(enumerate(iris.target_names)))
    return df


def split_features_target(df: pd.DataFrame, target_column: str):
    """Separate the feature matrix X from the target vector y."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def make_train_test_split(X, y, test_size: float, random_state: int):
    """Stratified train/test split so class balance is preserved."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
