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

# Preprocessing function (you can adjust based on your dataset)
def preprocess_data_in_chunks(file_bytes, chunk_size=1000):
    try:
        # Use a generator to read the file in chunks
        data_gen = pd.read_csv(io.BytesIO(file_bytes), chunksize=chunk_size)
        
        # Preprocess each chunk as it's read
        for chunk in data_gen:
            # Example: scale features or do other preprocessing on the chunk
            chunk = chunk[['Temperature', 'Milk_visibility', 'IUFL', 'EUFL', 'IUFR', 'EUFR', 'IURL', 'EURR']]  # Add necessary features here
            # Process the chunk (example: scaling, etc.)
            # Return the preprocessed chunk
            yield chunk
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
        
        # Read the file in chunks and process it
        data_bytes = await file.read()
        
        # Create an empty list to store the processed chunks
        processed_data = []

        # Iterate through chunks
        for chunk in preprocess_data_in_chunks(data_bytes):
            # Here, you can further process each chunk (e.g., scaling, reshaping)
            chunk_data = np.array(chunk)
            processed_data.append(chunk_data)

        # Once all chunks are processed, convert them to a numpy array or desired format
        final_data = np.vstack(processed_data)  # Example: stacking chunks vertically

        # Predict using the model
        prediction = model.predict(final_data)

        # Post-process the prediction if necessary
        predicted_class = np.argmax(prediction, axis=1)[0]  # Adjust this based on your model output
        
        return {"prediction": int(predicted_class)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
