"""
Tumor Growth Outcome Predictor — themed version with visuals and sound.

Run with:  streamlit run app.py
Folder must contain: app.py, tumor_model.pkl, assets/ (images + sounds)
"""

import streamlit as st
import joblib
import pandas as pd
import base64
from pathlib import Path

st.set_page_config(
    page_title="Tumor Growth Outcome Predictor",
    page_icon="🧬",
    layout="wide",
)

ASSETS = Path(__file__).parent / "assets"


# ---------------------------------------------------------------------------
# Helpers to embed local files directly into the page
# ---------------------------------------------------------------------------

def load_svg(filename):
    return (ASSETS / filename).read_text()


def audio_to_base64(filename):
    data = (ASSETS / filename).read_bytes()
    return base64.b64encode(data).decode()


# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #050a12 0%, #0a1420 100%);
    }
    .block-container {
        padding-top: 1rem;
    }
    h1, h2, h3, p, label, .stMarkdown {
        color: #e8f0f5 !important;
    }
    div[data-testid="stSlider"] label {
        color: #9fd3e8 !important;
        font-weight: 500;
    }
    .outcome-card {
        background: rgba(255,255,255,0.04);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
    }
    .param-card {
        background: rgba(255,255,255,0.03);
        border-radius: 14px;
        padding: 18px 22px;
        border: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Background ambient sound, looping and quiet, with a mute option
# ---------------------------------------------------------------------------

play_ambient = st.sidebar.toggle("Lab ambience sound", value=True)
st.sidebar.caption("Toggle the background hum on or off.")

if play_ambient:
    ambient_b64 = audio_to_base64("ambient_hum.wav")
    st.markdown(
        f"""
        <audio autoplay loop>
            <source src="data:audio/wav;base64,{ambient_b64}" type="audio/wav">
        </audio>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Header banner
# ---------------------------------------------------------------------------

st.markdown(load_svg("banner.svg"), unsafe_allow_html=True)

st.write("")
st.markdown(
    "<p style='text-align:center; color:#9fd3e8; font-size:16px;'>"
    "Adjust the biological parameters below to simulate a tumor's interaction "
    "with the immune system, then run the prediction to see the likely outcome."
    "</p>",
    unsafe_allow_html=True,
)
st.write("")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("tumor_model.pkl")

model = load_model()

# ---------------------------------------------------------------------------
# Parameter controls
# ---------------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="param-card">', unsafe_allow_html=True)
    mutation_rate = st.slider(
        "🧬 Mutation rate",
        min_value=0.0, max_value=0.2, value=0.05, step=0.01,
        help="How often dividing cancer cells gain resistance to immune attack."
    )
    immune_strength = st.slider(
        "🛡️ Immune strength",
        min_value=0.02, max_value=0.18, value=0.1, step=0.01,
        help="Base probability an immune cell kills a cancer cell on contact."
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="param-card">', unsafe_allow_html=True)
    initial_nutrient = st.slider(
        "🩸 Initial nutrient level",
        min_value=40, max_value=140, value=100, step=10,
        help="Starting nutrient available to cancer cells at simulation start."
    )
    nutrient_regen_rate = st.slider(
        "♻️ Nutrient regeneration rate",
        min_value=1, max_value=5, value=3, step=1,
        help="How much nutrient each patch regenerates per tick."
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
predict_clicked = st.button("🔬 Run prediction", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction + themed result card
# ---------------------------------------------------------------------------

if predict_clicked:
    chime_b64 = audio_to_base64("scan_chime.wav")
    st.markdown(
        f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{chime_b64}" type="audio/wav">
        </audio>
        """,
        unsafe_allow_html=True,
    )

    X = pd.DataFrame([{
        "mutation-rate": mutation_rate,
        "immune-strength": immune_strength,
        "initial-nutrient": initial_nutrient,
        "nutrient-regen-rate": nutrient_regen_rate,
    }])

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    classes = list(model.classes_)

    icon_map = {
        "contained": "icon_contained.svg",
        "stalemate": "icon_stalemate.svg",
        "aggressive": "icon_aggressive.svg",
    }
    label_map = {
        "contained": "Contained — the immune system clears the tumor",
        "stalemate": "Stalemate — tumor persists at low levels",
        "aggressive": "Aggressive growth — tumor overwhelms the immune response",
    }

    st.write("")
    result_col1, result_col2 = st.columns([1, 2])

    with result_col1:
        st.markdown(load_svg(icon_map[prediction]), unsafe_allow_html=True)

    with result_col2:
        st.markdown('<div class="outcome-card">', unsafe_allow_html=True)
        st.markdown(f"### {label_map[prediction]}")

        prob_df = pd.DataFrame({
            "Outcome": classes,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)

        st.bar_chart(prob_df.set_index("Outcome"))
        st.markdown('</div>', unsafe_allow_html=True)

st.write("")
st.markdown(
    "<p style='text-align:center; color:#5a7a8c; font-size:12px;'>"
    "Model trained on simulated NetLogo agent-based data. "
    "Predictions reflect patterns in simulated data only and have no clinical meaning."
    "</p>",
    unsafe_allow_html=True,
)