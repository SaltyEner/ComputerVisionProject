"""Streamlit demo: upload a leaf photo, get the predicted disease + Grad-CAM.

Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# allow `import src...` when run via `streamlit run`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import gradcam_overlay, load_model, predict  # noqa: E402


@st.cache_resource
def _get_model():
    return load_model()


st.set_page_config(page_title="LeafDoctor", page_icon="🌱", layout="centered")
st.title("🌱 LeafDoctor — Plant Disease Classifier")
st.caption(
    "Transfer learning (MobileNet/ResNet/EfficientNet) on PlantVillage, "
    "with Grad-CAM explainability."
)

try:
    lm = _get_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

uploaded = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload an image to see the prediction and the Grad-CAM heatmap.")
    st.stop()

img = Image.open(uploaded)
preds = predict(lm, img, top_k=5)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Input")
    st.image(img, use_column_width=True)
with col2:
    st.subheader("Grad-CAM")
    st.image(gradcam_overlay(lm, img), use_column_width=True)

st.subheader(f"Prediction: **{preds[0]['label']}**  ·  {preds[0]['probability']:.1%}")
st.bar_chart(
    {p["label"]: p["probability"] for p in preds},
)
