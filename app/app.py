"""Streamlit demo app.

A lightweight UI so anyone can interact with the trained model without
writing code. Run from the repo root:

    streamlit run app/app.py

Train a model first if you haven't:  python scripts/train.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow "from src import ..." when Streamlit runs this file from app/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import load_model
from src.utils import load_config

st.set_page_config(page_title="ML Project Demo", page_icon="🌸")
st.title("🌸 Iris Classifier Demo")
st.caption(
    "Example app for the ML project template. "
    "Replace the inputs below with the features of your own model."
)

config = load_config()

try:
    model = load_model(config["output"]["model_path"])
except FileNotFoundError:
    st.error("No trained model found. Run `python scripts/train.py` first.")
    st.stop()

st.subheader("Enter flower measurements")
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8)
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
with col2:
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.3)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.3)

# Column names must match the training data exactly.
X_new = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ],
)

prediction = model.predict(X_new)[0]
st.subheader("Prediction")
st.success(f"Predicted species: **{prediction}**")

if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X_new)[0]
    proba_df = pd.DataFrame({"species": model.classes_, "probability": proba})
    st.bar_chart(proba_df.set_index("species"))
