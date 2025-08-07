import os
import mlflow
from typing import Union, List
import pandas as pd
import numpy as np

# -------- CONFIG --------
EXPERIMENT_NAME = "goemotions-classification-2025-08-06"
MLRUNS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../mlartifacts"))
mlflow.set_tracking_uri(f"file://{MLRUNS_PATH}")

# -------- LABEL MAP --------
label_map = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval",
    5: "caring", 6: "confusion", 7: "curiosity", 8: "desire", 9: "disappointment",
    10: "disapproval", 11: "disgust", 12: "embarrassment", 13: "excitement",
    14: "fear", 15: "gratitude", 16: "grief", 17: "joy", 18: "love",
    19: "nervousness", 20: "optimism", 21: "pride", 22: "realization",
    23: "relief", 24: "remorse", 25: "sadness", 26: "surprise", 27: "neutral"
}

# -------- MODEL LOADING --------
def get_latest_run_id(experiment_name: str) -> str:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
    if runs.empty:
        raise ValueError(f"No runs found for experiment '{experiment_name}'.")

    return runs.loc[0, "run_id"]

def load_model(run_id: str):
    model_uri = f"runs:/{run_id}/model"
    print(f"🔁 Loading model from run ID: {run_id}")
    return mlflow.pyfunc.load_model(model_uri)

# -------- TEXT PREPROCESSING --------
def preprocess_text(text: str) -> str:
    return text.lower().strip()

# -------- PREDICTION --------
def predict_emotion(model, text: Union[str, List[str]], threshold: float = 0.5) -> List[List[str]]:
    if isinstance(text, str):
        text = [text]

    processed_text = [preprocess_text(t) for t in text]
    input_df = pd.DataFrame({"text": processed_text})

    # Get prediction probabilities or fallback to binary predictions
    try:
        probs = model.predict_proba(input_df)
        preds = (np.array(probs) >= threshold).astype(int)
    except AttributeError:
        preds = model.predict(input_df)

    # Ensure 2D shape for both single and batch inputs
    preds = np.atleast_2d(preds)

    # Map prediction vectors to emotion labels
    results = []
    for row in preds:
        emotion_ids = [i for i, val in enumerate(row) if val == 1]
        emotions = [label_map[i] for i in emotion_ids]
        results.append(emotions if emotions else ["neutral"])

    return results

# -------- MAIN TEST --------
if __name__ == "__main__":
    run_id = get_latest_run_id(EXPERIMENT_NAME)
    model = load_model(run_id)

    sample_texts = [
        "I am so happy today!",
        "I feel very sad and lonely.",
        "This is surprising news!",
        "You made me so angry and disappointed.",
        "Thanks a lot, I appreciate your help!"
    ]
    predictions = predict_emotion(model, sample_texts, threshold=0.4)

    for text, pred in zip(sample_texts, predictions):
        print(f"\nText: {text}\nPredicted emotions: {', '.join(pred)}")