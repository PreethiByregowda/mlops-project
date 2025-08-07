import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel
from typing import Union, List
import mlflow
from emotion_model import predict

# ---------------- CONFIG ----------------
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "goemotions-classification-2025-08-06"

# ---------------- LOAD MODEL ----------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")

# Get the latest run from the experiment
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
if runs.empty:
    raise ValueError(f"No runs found for experiment ID {experiment.experiment_id}")

latest_run_id = runs.loc[0, "run_id"]
print(f"🔁 Loading model from run ID: {latest_run_id}")
model = predict.load_model(latest_run_id)

# ---------------- FASTAPI APP ----------------
app = FastAPI(
    title="Emotion Recognition API",
    description="Predict emotions from text using a multi-label ML model.",
    version="1.0.0"
)

class TextRequest(BaseModel):
    text: Union[str, List[str]]
    threshold: float = 0.5  # Optional threshold override

@app.post("/predict")
def predict_endpoint(request: TextRequest = Body(...)):
    try:
        emotions = predict.predict_emotion(model, request.text, threshold=request.threshold)

        # Return format depends on input type
        if isinstance(request.text, str):
            return {"emotions": emotions[0]}  # Return single list of emotions
        else:
            return {"emotions": emotions}     # Return list of lists
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict")
def predict_get(
    text: str = Query(..., description="Text to analyze"),
    threshold: float = Query(0.5, description="Prediction threshold (0-1)")
):
    try:
        emotions = predict.predict_emotion(model, text, threshold=threshold)
        return {"emotions": emotions[0]}  # Always return a list of emotions
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))