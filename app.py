from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np
import pandas as pd
import io

app = FastAPI()

# Load the trained model when the FastAPI app starts
try:
    model = tf.keras.models.load_model("my_final_model.keras")
    print("Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Define feature columns (Ensure these match the trained model)
FEATURES = ['Temperature', 'Milk_visibility', 'IUFL', 'EUFL', 'IUFR', 'EUFR', 'IURL', 'EURR']
SEQ_LENGTH = 10  # Sequence length is 10

# Preprocessing function to process CSV in chunks
def preprocess_data(file_bytes):
    try:
        # Read CSV
        df = pd.read_csv(io.BytesIO(file_bytes))

        # Ensure the CSV contains required features
        if not all(col in df.columns for col in FEATURES):
            raise HTTPException(status_code=400, detail="CSV is missing required feature columns")

        # Scale features (normalization or standardization can be applied here)
        df = df[FEATURES]
        scaled_data = (df - df.mean()) / df.std()  # Standardization (adjust as needed)

        # Ensure enough data points for a sequence
        if len(scaled_data) < SEQ_LENGTH:
            raise HTTPException(status_code=400, detail=f"CSV must have at least {SEQ_LENGTH} rows for sequence input")

        # Use last SEQ_LENGTH rows to form a single sequence
        sequence_data = np.array([scaled_data[-SEQ_LENGTH:].values])  # Shape: (1, 10, 8)

        return sequence_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")

# Root route
@app.get("/")
def read_root():
    return {"message": "Welcome to the Mastitis Detection API!"}

# Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Only accept CSV files
        if not file.filename.lower().endswith("csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")

        # Read the uploaded CSV file
        data_bytes = await file.read()

        # Preprocess the CSV file
        processed_data = preprocess_data(data_bytes)  # Shape: (1, 10, 8)

        # Predict using the trained model
        prediction = model.predict(processed_data)

        # Convert prediction output (binary classification)
        predicted_class = int(np.round(prediction[0]))  # For binary classification (0 or 1)

        return {"prediction": predicted_class}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


