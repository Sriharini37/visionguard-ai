"""
VisionGuard AI — FastAPI model-serving layer.

Exposes a REST endpoint for defect inspection so the model can be integrated
into a factory MES/SCADA system, not just the Streamlit demo.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Then:
    curl -X POST "http://localhost:8000/inspect" -F "file=@sample.jpg"
"""
import io
import os
import time
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

app = FastAPI(
    title="VisionGuard AI API",
    description="Model-serving layer for autonomous defect detection",
    version="1.0.0",
)

MODEL_PATH = os.environ.get("VISIONGUARD_WEIGHTS", "runs/detect/visionguard_yolov8/weights/best.pt")
_model = None


def get_model():
    """Lazy-load the YOLOv8 model on first request. Returns None if weights
    aren't available yet, so the API can still boot and serve /health."""
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        from ultralytics import YOLO
        _model = YOLO(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    model = get_model()
    return {"status": "ok", "model_loaded": model is not None, "model_path": MODEL_PATH}


@app.post("/inspect")
async def inspect(file: UploadFile = File(...), confidence: float = 0.4):
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"No model weights found at {MODEL_PATH}. Train the model or set "
                    "VISIONGUARD_WEIGHTS to a valid checkpoint.",
        )

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    t0 = time.time()
    results = model.predict(np.array(img), conf=confidence, verbose=False)
    latency_ms = (time.time() - t0) * 1000

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": r.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox_xyxy": [float(v) for v in box.xyxy[0]],
            })

    verdict = "DEFECT" if detections else "PASS"

    return JSONResponse({
        "verdict": verdict,
        "detections": detections,
        "latency_ms": round(latency_ms, 2),
        "model_path": MODEL_PATH,
    })


@app.get("/")
def root():
    return {
        "service": "VisionGuard AI",
        "endpoints": ["/health", "/inspect (POST, multipart image)"],
        "docs": "/docs",
    }
