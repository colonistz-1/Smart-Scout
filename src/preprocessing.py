
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "scoring.expected_goals",
    "scoring.expected_assists",
    "defending.tackles",
    "defending.interceptions",
]

MODEL_FEATURES = [f"{feature}_per90" for feature in FEATURES]


def preprocess_data(df, min_minutes=450):
    df = df.copy()

    required = ["position", "minutes_played", *FEATURES]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Convert required numeric columns
    for col in ["minutes_played", *FEATURES]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter eligible midfielders
    df = df[
        (df["position"] == "M")
        & (df["minutes_played"] >= min_minutes)
    ].copy()

    # Calculate per-90 metrics
    for feature in FEATURES:
        df[f"{feature}_per90"] = (
            df[feature] / df["minutes_played"] * 90
        )

    # Exclude incomplete records
    df = df.dropna(subset=MODEL_FEATURES)

    return df


def scale_features(df):
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(df[MODEL_FEATURES])

    return X_scaled, scaler
