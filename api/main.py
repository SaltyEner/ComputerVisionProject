"""FastAPI service exposing the classifier over HTTP.

Run with:
    uvicorn api.main:app --reload

Endpoints:
    GET  /health        -> {"status": "ok", "classes": N}
    POST /predict       -> top-k predictions for an uploaded image
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import load_model, predict  # noqa: E402

app = FastAPI(title="LeafDoctor API", version="1.0.0")
_model = None  # lazily loaded so the import never crashes without a checkpoint


def _ensure_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


@app.get("/health")
def health() -> dict:
    try:
        lm = _ensure_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok", "classes": len(lm.classes), "backbone": lm.backbone}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...), top_k: int = 5) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    try:
        lm = _ensure_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid image.") from exc

    return {"predictions": predict(lm, img, top_k=top_k)}
