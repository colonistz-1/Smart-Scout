"""Streamlit demo app.

A lightweight UI so anyone can interact with the trained model without
writing code. Run from the repo root:

    streamlit run app/app.py

Train a model first if you haven't:  python scripts/train.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import load_model
from src.similarity import find_similar

st.set_page_config(
    page_title="Smart-Scout",
    page_icon="app/assets/soccer-ball_26bd.png",
    layout="wide",
)

FEATURES = [
    "scoring.expected_goals_per90",
    "scoring.expected_assists_per90",
    "defending.tackles_per90",
    "defending.interceptions_per90",
    "passing.key_passes_per90",
    "passing.accurate_crosses_per90",
]

LABELS = {
    "scoring.expected_goals_per90": "xG / 90",
    "scoring.expected_assists_per90": "xA / 90",
    "defending.tackles_per90": "Tackles / 90",
    "defending.interceptions_per90": "Interceptions / 90",
    "passing.key_passes_per90": "Key passes / 90",
    "passing.accurate_crosses_per90": "Accurate crosses / 90",
}

CLUSTER_NAMES = {
    0: "Creator",
    1: "Defensive Minded",
    2: "Goal Threat",
}

@st.cache_resource
def load_bundle():
    return load_model(ROOT / "models/kmeans_v2.joblib")


@st.cache_data
def load_players():
    return pd.read_csv(
        ROOT / "data/processed/midfielder_clusters_v2.csv"
    )


st.title("Smart-Scout")
st.caption(
    "Explore statistical similarities between Premier League midfielders."
)

try:
    bundle = load_bundle()
    df = load_players()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Train your v2 model before starting the app.")
    st.stop()

scaler = bundle["scaler"]

if hasattr(scaler, "feature_names_in_"):
    features = list(scaler.feature_names_in_)
else:
    features = list(bundle.get("features", FEATURES))

required = ["player_name", "minutes_played", "cluster", *features]
missing = [column for column in required if column not in df.columns]

if missing:
    st.error(f"Missing dataset columns: {missing}")
    st.stop()

if len(df) < 2:
    st.error("At least two players are needed for similarity search.")
    st.stop()

# This version identifies players by their unique names.
if df["player_name"].isna().any() or df["player_name"].duplicated().any():
    st.error("Player names must be present and unique for this version.")
    st.stop()

with st.sidebar:
    st.header("Scouting search")

    player_name = st.selectbox(
        "Select a midfielder",
        sorted(df["player_name"].tolist()),
    )

    max_matches = min(10, len(df) - 1)
    top_n = st.selectbox(
        "Number of players",
        list(range(1, max_matches + 1)),
        index=min(5, max_matches) - 1,
    )

    st.caption(
        "Model: K-Means v2 · 3 clusters · "
        "6 per-90 features · Minimum 450 minutes"
    )

selected = df.loc[df["player_name"].eq(player_name)].iloc[0]

st.subheader(player_name)

col1, col2 = st.columns(2)
col1.metric("Minutes played", f"{int(selected['minutes_played']):,}")
col2.metric(
    "Statistical profile",
    CLUSTER_NAMES.get(int(selected["cluster"]), "Unlabelled"),
)

try:
    similar_players = find_similar(
        df=df,
        scaler=scaler,
        features=features,
        player_name=player_name,
        top_n=top_n,
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Similar players")
st.caption(
    "Lower distance means a closer profile across the standardized "
    "features. Results include players from all clusters."
)

results_table = similar_players[
    ["player_name", "minutes_played", "distance", *features]
].rename(
    columns={
        "player_name": "Player",
        "minutes_played": "Minutes",
        "distance": "Distance",
        **LABELS,
    }
)

st.dataframe(
    results_table.round(3),
    hide_index=True,
    width="stretch",
)

st.subheader("Compare players")

comparison_options = sorted(
    df.loc[
        df["player_name"] != player_name,
        "player_name",
    ].tolist()
)

# Default to the closest match, but allow any other player.
default_comparison = similar_players.iloc[0]["player_name"]

comparison_name = st.selectbox(
    "Search for a player to compare",
    options=comparison_options,
    index=comparison_options.index(default_comparison),
    key=f"comparison_{player_name}",
)

comparison = (
    df.set_index("player_name")
    .loc[[player_name, comparison_name], features]
)

st.write("**Per-90 statistics**")
st.dataframe(
    comparison.rename(columns=LABELS).T.round(3),
    width="stretch",
)

# Calculate percentiles against the full dataset, not just these two players.
percentiles = df[features].rank(
    method="average",
    pct=True,
) * 100

percentiles.index = df["player_name"]

player_names = [player_name, comparison_name]
radar_values = percentiles.loc[player_names, features]

# Display the percentile values above the chart.
st.subheader("Player comparison")

# Rank against every eligible midfielder, not just the selected pair.
percentiles = df[features].rank(
    method="average",
    pct=True,
) * 100

percentiles.index = df["player_name"]

radar_data = (
    percentiles.loc[[player_name, comparison_name]]
    .rename(columns=LABELS)
    .rename_axis("Player")
    .reset_index()
    .melt(
        id_vars="Player",
        var_name="Metric",
        value_name="Percentile",
    )
)

fig = px.line_polar(
    radar_data,
    r="Percentile",
    theta="Metric",
    color="Player",
    line_close=True,
    range_r=[0, 100],
    color_discrete_map={
        player_name: "#1195EC",
        comparison_name: "#50FC58",
    },
    markers=True,
)

fig.update_traces(
    fill="toself",
    line={"width": 3},
    hovertemplate=(
        "%{theta}<br>Percentile: %{r:.1f}"
        "<extra>%{fullData.name}</extra>"
    ),
)

fig.update_layout(
    template=None,
    height=600,
    font={"size": 14},
    polar={
        "bgcolor": "rgba(0,0,0,0)",
        "angularaxis": {
            "tickfont": {"size": 15, "color": None},
        },
        "radialaxis": {
            "range": [0,100],
            "tickvals": [25, 50, 75, 100],
            "gridcolor": "#444444",
        },
    },
    legend={
        "orientation": "h",
        "y": -0.15,
        "x": 0.5,
        "xanchor": "center",
    },
    margin={"l": 80, "r": 80, "t": 60, "b": 80},
)

st.plotly_chart(fig, use_container_width=True, theme="streamlit")

with st.expander("Method and limitations"):
    st.write(
        "Similarity uses Euclidean distance across six features, transformed "
        "with the scaler saved during training. The app does not retrain "
        "the model."
    )
    st.write(
        "Profiles describe statistical contributions rather than fixed "
        "positions or player quality. Team tactics, possession, set-piece "
        "duties, and playing time influence the results. Similarity does "
        "not predict transfer success."
    )