from __future__ import annotations

import os
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated

import mlflow
import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from food11 import data

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_URI = "models:/food11@champion"

# Same class order ImageFolder assigns at training time: sorted class-folder names.
CLASSES = sorted(data.CLASSES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

model: mlflow.pyfunc.PyFuncModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    # SKIP_MODEL lets unit tests and the CI smoke test start the app with no MLflow server;
    # tests then swap a fake into the module-level `model`.
    if not os.environ.get("SKIP_MODEL"):
        mlflow.set_tracking_uri(TRACKING_URI)
        model = mlflow.pyfunc.load_model(MODEL_URI)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File()]) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).numpy()

    logits = np.asarray(model.predict(tensor))
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    predicted_idx = int(probabilities.argmax(axis=1)[0])

    return {
        "category": CLASSES[predicted_idx],
        "confidence": float(probabilities[0, predicted_idx]),
    }
