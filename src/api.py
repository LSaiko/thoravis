"""
thoravis/src/api.py
─────────────────────────────────────────────────────────────────────────────
FastAPI inference service — upload a chest X-ray, get back per-pathology
probabilities and/or a Grad-CAM overlay for one pathology.

Run
---
    uvicorn src.api:app --host 0.0.0.0 --port 8000

Config (env vars)
------------------
    THORAVIS_CHECKPOINT  path to a trained checkpoint (default: models/best_thoravis.pt)
    THORAVIS_VIT_CKPT    override the ViT backbone (default: google/vit-base-patch16-224-in21k;
                          tests point this at a tiny local checkpoint to stay offline/fast)

Endpoints
---------
    GET  /health                     liveness + which device/checkpoint is loaded
    POST /predict                    -> JSON: {pathology: probability, ...}
    POST /predict/gradcam?label=...  -> PNG Grad-CAM overlay for one pathology
"""

import io
import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from src.dataset import PATHOLOGY_LABELS
from src.gradcam import GradCAMViT
from src.model import ThoraVisClassifier
from src.preprocessing import XRayPreprocessor, overlay_heatmap

CHECKPOINT_PATH = os.environ.get("THORAVIS_CHECKPOINT", "models/best_thoravis.pt")
VIT_CKPT = os.environ.get("THORAVIS_VIT_CKPT", ThoraVisClassifier.VIT_CKPT)

state = {}  # holds "device", "model" — set at startup, read by request handlers


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.isfile(CHECKPOINT_PATH):
        raise RuntimeError(
            f"No checkpoint at '{CHECKPOINT_PATH}'. Train one first "
            f"(python -m src.train) or set THORAVIS_CHECKPOINT."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThoraVisClassifier(pretrained_ckpt=VIT_CKPT).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    state["device"] = device
    state["model"] = model
    state["preprocessor"] = XRayPreprocessor(augment=False)
    state["checkpoint_epoch"] = checkpoint.get("epoch")
    yield
    state.clear()


app = FastAPI(title="ThoraVis Inference API", version="1.0.0", lifespan=lifespan)


def _read_image(upload: UploadFile) -> Image.Image:
    try:
        return Image.open(io.BytesIO(upload.file.read()))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail=f"Not a readable image: {exc}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(state["device"]),
        "checkpoint": CHECKPOINT_PATH,
        "checkpoint_epoch": state["checkpoint_epoch"],
    }


@app.post("/predict")
def predict(file: UploadFile = File(...), threshold: float = Query(0.5, ge=0.0, le=1.0)):
    pil_img = _read_image(file)
    image_tensor = state["preprocessor"].preprocess_pil(pil_img).unsqueeze(0).to(state["device"])

    probs = state["model"].predict_proba(image_tensor)[0]
    predictions = {label: float(p) for label, p in zip(PATHOLOGY_LABELS, probs)}
    above_threshold = {k: v for k, v in predictions.items() if v >= threshold}

    return {"predictions": predictions, "above_threshold": above_threshold, "threshold": threshold}


@app.post("/predict/gradcam")
def predict_gradcam(file: UploadFile = File(...), label: str = Query(...)):
    if label not in PATHOLOGY_LABELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown label '{label}'. Choose from {PATHOLOGY_LABELS}",
        )
    pil_img = _read_image(file)
    image_tensor = state["preprocessor"].preprocess_pil(pil_img).unsqueeze(0).to(state["device"])

    cam = GradCAMViT(state["model"], device=state["device"])
    try:
        heatmap = cam.generate(image_tensor, target_class=PATHOLOGY_LABELS.index(label))
    finally:
        cam.remove_hooks()

    original = np.array(pil_img.convert("RGB").resize((224, 224)))
    blended = overlay_heatmap(original, heatmap)

    ok, png_bytes = cv2.imencode(".png", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode Grad-CAM overlay")
    return Response(content=png_bytes.tobytes(), media_type="image/png")
