# ------------------------------------------------------------
# serve.py - A small web API that serves the food11 model
#
# GET  /health   -> checks that the API is running
# POST /predict  -> receives an image, returns the food category
# ------------------------------------------------------------

# io lets us treat the uploaded bytes like a file
import io
# os lets us read environment variables
import os
# asynccontextmanager lets us run code once when the API starts
from contextlib import asynccontextmanager

import mlflow
import mlflow.pyfunc
import numpy as np
# FastAPI: the web framework. UploadFile/File: to receive uploaded files.
# HTTPException: to send back a clear error message
from fastapi import FastAPI, File, HTTPException, UploadFile
# PIL opens image files (jpg, png...)
from PIL import Image
# The same image preparation tools we used in train.py
from torchvision import transforms


# ------------------------------------------------------------
# Settings (read from environment variables, with defaults)
# ------------------------------------------------------------

# Where the mlflow server is.
# On the laptop the default works. Inside a container we will change it
# with an environment variable, without touching the code.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# Which model to load: "the version of food11 that has the champion alias"
MODEL_URI = os.getenv("MODEL_URI", "models:/food11@champion")

# The 11 categories, in the same order as during training
# (ImageFolder sorted the folder names alphabetically)
CLASSES = [
    "Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat",
    "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit",
]

# Prepare an image exactly like in training:
# resize to 128x128, turn it into numbers, adjust the colors like ResNet expects
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# The model will be stored here once it is loaded
model = None


# ------------------------------------------------------------
# Load the model ONCE, when the API starts
# (not at every request, that would be very slow)
# ------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    mlflow.set_tracking_uri(TRACKING_URI)
    print(f"Loading model {MODEL_URI} from {TRACKING_URI} ...")
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print("Model loaded.")
    yield  # the API runs here, until it is stopped


app = FastAPI(title="Food-11 classifier API", lifespan=lifespan)

# ------------------------------------------------------------
# Helper: turn the model's raw scores into probabilities
# ------------------------------------------------------------
def softmax(scores):
    # Subtracting the max keeps the numbers small (avoids overflow)
    exp = np.exp(scores - np.max(scores))
    return exp / exp.sum()


# ------------------------------------------------------------
# GET /health -> is the API alive?
# ------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ------------------------------------------------------------
# POST /predict -> receive an image, return the food category
# ------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read the uploaded file and open it as an image
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    # 2. Prepare it like in training, and add a "batch" dimension:
    #    [3, 128, 128] -> [1, 3, 128, 128]  (a batch of 1 image)
    batch = preprocess(image).unsqueeze(0).numpy().astype(np.float32)

    # 3. Ask the model: it returns 11 raw scores for our 1 image
    scores = np.asarray(model.predict(batch))[0]

    # 4. Turn scores into probabilities and pick the highest one
    probabilities = softmax(scores)
    best = int(np.argmax(probabilities))

    return {
        "filename": file.filename,
        "category": CLASSES[best],
        "class_id": best,
        "confidence": round(float(probabilities[best]), 4),
    }