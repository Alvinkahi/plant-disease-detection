from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance
import io
import os
import json
from datetime import datetime
from rembg import remove  

app = FastAPI()

# -------- Paths --------
MODEL_PATH = "dataset.h5"
CLASS_NAMES_PATH = "class_names.json"
LOG_FILE = "predictions_log.json"
LOG_IMAGE_DIR = "logs"

# Create logs folder if it doesn't exist
os.makedirs(LOG_IMAGE_DIR, exist_ok=True)

# -------- Load Class Names --------
if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"Class names file not found at {CLASS_NAMES_PATH}")

with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

# -------- Prevention & Cure Advice --------
DISEASE_INFO = {
    "Healthy": {
        "prevention": "Maintain good irrigation, proper spacing, and balanced fertilization.",
        "cure": "No cure needed. Continue with regular crop care practices."
    },
    "Rust": {
        "prevention": "Use resistant varieties, avoid overhead watering, and ensure good airflow.",
        "cure": "Spray with Mancozeb or Copper fungicides. Remove infected leaves."
    },
    "Powdery": {
        "prevention": "Plant in sunny areas, avoid excess nitrogen, and water at the base.",
        "cure": "Use sulfur fungicides or potassium bicarbonate sprays. Prune affected parts."
    }
}

# -------- Load Model --------
print(f" Loading model from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f" Model file not found at {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print(" Model loaded successfully!")

# -------- Preprocessing Function --------
def preprocess_image(image_data: bytes):
    try:
        #  Background removal (works with bytes, not PIL.Image)
        img_no_bg_bytes = remove(image_data)
        img = Image.open(io.BytesIO(img_no_bg_bytes)).convert("RGB")
    except Exception as e:
        print(f" Background removal failed: {e}")
        img = Image.open(io.BytesIO(image_data)).convert("RGB")

    #  Lighting correction
    enhancer = ImageEnhance.Brightness(img)
    img_bright = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Contrast(img_bright)
    img_corrected = enhancer.enhance(1.2)

    #  Resize to model input size
    img_resized = img_corrected.resize((224, 224))

    #  Normalize
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# -------- Prediction Function --------
def predict_image(image_data: bytes):
    img_array = preprocess_image(image_data)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(predictions[0][predicted_index]) * 100

    return {
        "prediction": predicted_label,
        "confidence": round(confidence, 2),
        "prevention": DISEASE_INFO[predicted_label]["prevention"],
        "cure": DISEASE_INFO[predicted_label]["cure"]
    }

# -------- Save Prediction Log --------
def save_prediction_log(result, filename, image_bytes):
    img_path = os.path.join(LOG_IMAGE_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
    with open(img_path, "wb") as img_file:
        img_file.write(image_bytes)

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file": filename,
        "saved_image": img_path,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "prevention": result["prevention"],
        "cure": result["cure"]
    }

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)

    print(f" Saved prediction log for {filename}")

# -------- API Routes --------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = predict_image(contents)

        # Save logs
        save_prediction_log(result, file.filename, contents)

        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"message": " Crop Disease Detection API with preprocessing is running!"}
