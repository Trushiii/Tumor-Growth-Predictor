"""
Streamlit app for the Tumor Growth Outcome Predictor.

Loads the trained model (tumor_model.pkl) and lets the user adjust
the four simulation parameters with sliders to see a live prediction
of the tumor outcome category.

Run with:  streamlit run app.py
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Tumor Growth Outcome Predictor", page_icon="🧬")

# ---------------------------------------------------------------------------
# Load the trained model
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("tumor_model.pkl")

model = load_model()

# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.title("Tumor Growth Outcome Predictor")
st.write(
    "This tool predicts the likely outcome of a simulated tumor "
    "(contained, stalemate, or aggressive growth) based on four "
    "biological parameters from our NetLogo agent-based model."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    mutation_rate = st.slider(
        "Mutation Rate",
        min_value=0.0, max_value=0.2, value=0.05, step=0.01,
        help="How often dividing cancer cells gain resistance to immune attack."
    )
    immune_strength = st.slider(
        "Immune Strength",
        min_value=0.02, max_value=0.18, value=0.1, step=0.01,
        help="Base probability an immune cell kills a cancer cell on contact."
    )

with col2:
    initial_nutrient = st.slider(
        "Initial Nutrient Level",
        min_value=40, max_value=140, value=100, step=10,
        help="Starting nutrient available to cancer cells at simulation start."
    )
    nutrient_regen_rate = st.slider(
        "Nutrient Regeneration Rate",
        min_value=1, max_value=5, value=3, step=1,
        help="How much nutrient each patch regenerates per tick."
    )

st.divider()

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

if st.button("Predict Outcome", type="primary"):
    X = pd.DataFrame([{
        "mutation-rate": mutation_rate,
        "immune-strength": immune_strength,
        "initial-nutrient": initial_nutrient,
        "nutrient-regen-rate": nutrient_regen_rate,
    }])

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    classes = model.classes_

    outcome_colors = {
        "contained": "green",
        "stalemate": "orange",
        "aggressive": "red",
    }
    color = outcome_colors.get(prediction, "blue")

    st.markdown(f"### Predicted outcome: :{color}[{prediction.upper()}]")

    st.write("Prediction confidence by category:")
    prob_df = pd.DataFrame({
        "Outcome": classes,
        "Probability": probabilities
    }).sort_values("Probability", ascending=False)

    st.bar_chart(prob_df.set_index("Outcome"))

st.divider()
st.caption(
    "Model trained on simulated data from a NetLogo agent-based model of "
    "tumor growth under immune system pressure. Predictions reflect patterns "
    "learned from simulated data only and have no clinical meaning."
)
