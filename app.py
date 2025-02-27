from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load your trained model
model = tf.keras.models.load_model("my_final_model.keras")  # Change to your model filename

# Preprocessing function
def preprocess_image(image) -> np.array:
    image = Image.open(io.BytesIO(image))
    image = image.resize((224, 224))  # Adjust based on your model input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = preprocess_image(image_bytes)
    
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Assuming classification
    
    return {"prediction": int(predicted_class)}

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}