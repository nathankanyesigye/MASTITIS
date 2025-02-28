from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import io

app = FastAPI()

# Load the trained model
try:
    model = tf.keras.models.load_model("my_final_model.keras")  # Ensure this file exists
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Preprocessing function for tabular data
def preprocess_data(data_bytes) -> np.array:
    try:
        # Load CSV data
        data = pd.read_csv(io.BytesIO(data_bytes))
        
        # Select the features you need (assuming the columns match your trained model)
        features = data[['IUFL', 'EUFL', 'IUFR', 'EUFR', 'IURL', 'EUFR', 'Temperature', 'Milk_visibility']]  # Add correct feature columns
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(features)
        
        return np.array(scaled_data)  # Return the processed data
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file format or data")

# Root route
@app.get("/")
def read_root():
    return {"message": "Welcome to the Mastitis Detection API!"}

# Prediction endpoint for tabular data (CSV)
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Check file type
    if not file.filename.lower().endswith("csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    data_bytes = await file.read()
    processed_data = preprocess_data(data_bytes)
    
    # Assuming your model is for binary classification
    prediction = model.predict(processed_data)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Assuming classification
    
    return {"prediction": int(predicted_class)}

