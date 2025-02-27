from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load the trained model
try:
    model = tf.keras.models.load_model("my_final_model.keras")  # Ensure this file exists
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Preprocessing function
def preprocess_image(image_bytes) -> np.array:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Convert to RGB if needed
        image = image.resize((224, 224))  # Adjust based on your model input size
        image = np.array(image) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image format")

# Root route
@app.get("/")
def read_root():
    return {"message": "Welcome to the Mastitis Detection API!"}

# Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Check file type
    if not file.filename.lower().endswith(("png", "jpg", "jpeg")):
        raise HTTPException(status_code=400, detail="Only PNG, JPG, and JPEG files are allowed")

    image_bytes = await file.read()
    image = preprocess_image(image_bytes)
    
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Assuming classification
    
    return {"prediction": int(predicted_class)}
