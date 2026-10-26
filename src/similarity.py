import numpy as np

def find_similar(df, scaler, features, player_name, top_n=5):
    if hasattr(scaler, "feature_names_in_"):
        features = list(scaler.feature_names_in_)
    
    matches = np.flatnonzero(
    df["player_name"].eq(player_name).to_numpy())   

    if len(matches) != 1:
        raise ValueError(
            f"Expected one match for {player_name!r}, found {len(matches)}."
    )

    player_position = matches[0]
    X_scaled = scaler.transform(df[features])

    distances = np.linalg.norm(
        X_scaled - X_scaled[player_position],
        axis=1,
    )

    results = df.copy()
    results["distance"] = distances

    # Exclude the selected player by row position.
    results = results.iloc[
        np.arange(len(results)) != player_position
    ]

    return results.sort_values("distance").head(top_n)

