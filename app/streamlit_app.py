"""LeafDoctor — Streamlit demo.

A polished UI to upload (or pick) a leaf photo and get the predicted disease
plus a Grad-CAM explanation.

Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# allow `import src...` when launched via `streamlit run`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.inference import gradcam_overlay, load_model, predict  # noqa: E402

SAMPLES_DIR = ROOT / "samples"

st.set_page_config(page_title="LeafDoctor", page_icon="🌱", layout="wide")

# --- light cosmetic styling -------------------------------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1100px;}
      #MainMenu, footer {visibility: hidden;}
      .hero-title {font-size: 2.4rem; font-weight: 800; margin-bottom: .1rem;}
      .hero-sub   {color: #6b7280; font-size: 1.05rem; margin-bottom: 1.2rem;}
      .result-card {border-radius: 16px; padding: 1.1rem 1.3rem; margin-top:.3rem;
                    border: 1px solid rgba(0,0,0,.08);}
      .pill {display:inline-block; padding:.18rem .7rem; border-radius:999px;
             font-weight:700; font-size:.85rem;}
      .pill-ok  {background:#dcfce7; color:#166534;}
      .pill-bad {background:#fee2e2; color:#991b1b;}
      .big-label {font-size:1.6rem; font-weight:800; margin:.35rem 0 .1rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_model()


def read_val_acc() -> float | None:
    try:
        data = json.loads((config.ARTIFACTS_DIR / "metrics.json").read_text())
        return data.get("best_val_acc")
    except Exception:  # noqa: BLE001
        return None


# --- sidebar ----------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌱 LeafDoctor")
    st.caption("Plant-disease classifier with explainable AI.")
    st.divider()
    try:
        _lm = get_model()
        st.markdown("**Model**")
        st.write(f"- Backbone: `{_lm.backbone}`")
        st.write(f"- Classes: `{len(_lm.classes)}`")
        acc = read_val_acc()
        if acc is not None:
            st.write(f"- Val accuracy: `{acc:.1%}`")
    except FileNotFoundError:
        _lm = None
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. A pretrained CNN extracts visual features.\n"
        "2. A classifier predicts the disease.\n"
        "3. **Grad-CAM** highlights the regions that drove the decision."
    )
    st.divider()
    st.markdown(
        "[⭐ Source on GitHub]"
        "(https://github.com/SaltyEner/ComputerVisionProject)"
    )


# --- header -----------------------------------------------------------------
st.markdown('<div class="hero-title">🌱 LeafDoctor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Upload a leaf photo — get the predicted disease and a '
    "heatmap showing <i>why</i>.</div>",
    unsafe_allow_html=True,
)

if _lm is None:
    st.error(
        "No trained model found. Train it first with `python -m src.train` "
        "(or run the Colab notebook and drop `model.pt` into `artifacts/`)."
    )
    st.stop()
lm = _lm

# --- pick an image ----------------------------------------------------------
samples = sorted(SAMPLES_DIR.glob("*.jpg")) if SAMPLES_DIR.exists() else []
tab_upload, tab_sample = st.tabs(["📤 Upload", "🖼️ Try a sample"])

img: Image.Image | None = None
with tab_upload:
    uploaded = st.file_uploader(
        "Choose a leaf image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )
    if uploaded is not None:
        img = Image.open(uploaded)
with tab_sample:
    if samples:
        names = [p.stem.replace("_", " ").title() for p in samples]
        choice = st.selectbox("Pick a sample leaf", names, label_visibility="collapsed")
        if st.button("Use this sample", type="primary"):
            img = Image.open(samples[names.index(choice)])
    else:
        st.info("No samples found in the `samples/` folder.")

if img is None:
    st.info("⬆️ Upload an image or pick a sample to see the prediction.")
    st.stop()

# --- inference --------------------------------------------------------------
with st.spinner("Analysing the leaf..."):
    preds = predict(lm, img, top_k=5)
    overlay = gradcam_overlay(lm, img)

top = preds[0]
healthy = "healthy" in top["label"].lower()
pill = '<span class="pill pill-ok">✅ Healthy</span>' if healthy \
    else '<span class="pill pill-bad">⚠️ Diseased</span>'

img_col, cam_col = st.columns(2)
with img_col:
    st.image(img, caption="Input leaf", use_container_width=True)
with cam_col:
    st.image(overlay, caption="Grad-CAM — what the model looked at",
             use_container_width=True)

st.markdown(
    f'<div class="result-card">{pill}'
    f'<div class="big-label">{top["label"]}</div>'
    f'<div style="color:#6b7280;">Confidence: <b>{top["probability"]:.1%}</b></div></div>',
    unsafe_allow_html=True,
)

st.markdown("#### Top predictions")
for p in preds:
    st.write(f"**{p['label']}** — {p['probability']:.1%}")
    st.progress(min(1.0, float(p["probability"])))

with st.expander("ℹ️ What is Grad-CAM?"):
    st.write(
        "Grad-CAM produces a heatmap over the image showing which regions most "
        "influenced the prediction. If the warm (red) area sits on the lesion "
        "rather than the background, the model is reasoning about the right thing "
        "— a quick visual trust check."
    )
