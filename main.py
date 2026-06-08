
"""
Deepfake Detector — FastAPI backend.

Run:
    uvicorn main:app --reload

then open:
    http://127.0.0.1:8000
"""

import io
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input


MODEL_PATH = "deepfake_finetuned.keras"
IMG_SIZE = (224, 224)

# Load model once
model = load_model(MODEL_PATH)

app = FastAPI(title="Deepfake Detector")


def predict_array(img):
    """Run the model on a PIL image and return sigmoid output."""

    img = img.convert("RGB").resize(IMG_SIZE)

    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    arr = preprocess_input(arr)

    prediction = model.predict(arr, verbose=0)

    return float(prediction[0][0])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    data = await file.read()

    try:
        img = Image.open(io.BytesIO(data))

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read the image."
        )

    raw = predict_array(img)

    if raw > 0.5:
        label = "REAL"
        confidence = raw * 100.0

    else:
        label = "FAKE"
        confidence = (1.0 - raw) * 100.0

    return JSONResponse({
        "label": label,
        "confidence": round(confidence, 2),
        "raw": round(raw, 4)
    })


@app.get("/")
async def index():
    return FileResponse("static/index.html")


# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Serve static files

app.mount("/static", StaticFiles(directory="static"), name="static")
