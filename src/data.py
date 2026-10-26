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
