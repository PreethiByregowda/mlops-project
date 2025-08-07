import time
from datetime import datetime
import mlflow
import mlflow.sklearn
from prefect import flow, task, get_run_logger
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from prefect.task_runners import SequentialTaskRunner
from collections import Counter

import pandas as pd
import ast


@task(name="Run models")
def run_models(X_train, y_train, X_valid, y_valid):
    for model_class in (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier):
        with mlflow.start_run():
            # Setup model pipeline with TF-IDF and model
            if model_class == LogisticRegression:
                model = make_pipeline(
                    TfidfVectorizer(),
                    model_class(max_iter=1000, class_weight='balanced', random_state=42)
                )
            else:
                model = make_pipeline(
                    TfidfVectorizer(),
                    model_class(random_state=42)
                )

            # Remove rows where target is NaN
            mask = ~y_train.isna()
            X_train = X_train[mask]
            y_train = y_train[mask]

            mask_valid = ~y_valid.isna()
            X_valid = X_valid[mask_valid]
            y_valid = y_valid[mask_valid]

            # Train the model
            model.fit(X_train["text"], y_train)

            # Measure inference time on train+valid data
            start_time = time.time()
            y_pred_train = model.predict(X_train["text"])
            y_pred_valid = model.predict(X_valid["text"])
            inference_time = time.time() - start_time

            # Log prediction distribution on validation set
            pred_dist = Counter(y_pred_valid)
            for label, count in pred_dist.items():
                mlflow.log_metric(f"pred_count_label_{label}", count)

            # Calculate and log metrics for train and valid sets
            metrics = {
                "accuracy_train": accuracy_score(y_train, y_pred_train),
                "accuracy_valid": accuracy_score(y_valid, y_pred_valid),
                "f1_train": f1_score(y_train, y_pred_train, average="weighted"),
                "f1_valid": f1_score(y_valid, y_pred_valid, average="weighted"),
                "precision_train": precision_score(y_train, y_pred_train, average="weighted"),
                "precision_valid": precision_score(y_valid, y_pred_valid, average="weighted"),
                "recall_train": recall_score(y_train, y_pred_train, average="weighted"),
                "recall_valid": recall_score(y_valid, y_pred_valid, average="weighted"),
            }

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log tags for author and model type
            mlflow.set_tag("author/developer", "PreethiB")
            mlflow.set_tag("Model", model_class.__name__)

            # Log inference time per sample (train + valid samples)
            total_samples = len(y_pred_train) + len(y_pred_valid)
            mlflow.log_metric("inference_time_per_sample", inference_time / total_samples)

            # Log the trained model artifact
            mlflow.sklearn.log_model(model, artifact_path="model")

import re

def safe_literal_eval(s):
    # Replace spaces between digits with commas (only inside brackets)
    # Example: "[8 20]" -> "[8,20]"
    if isinstance(s, str):
        s_fixed = re.sub(r'(\d)\s+(\d)', r'\1,\2', s)
        try:
            return ast.literal_eval(s_fixed)
        except Exception as e:
            print(f"Failed to parse: {s_fixed} with error: {e}")
            return None
    return None

def process_data():
    # Read CSV files
    df_train = pd.read_csv("data/train.csv")
    df_valid = pd.read_csv("data/valid.csv")

    # Convert string representations of lists to actual lists
    df_train['labels'] = df_train['labels'].apply(safe_literal_eval)
    df_valid['labels'] = df_valid['labels'].apply(safe_literal_eval)

    def get_first_label(labels):
        # Safely get first label if list is not empty
        if isinstance(labels, (list, tuple)) and len(labels) > 0:
            return labels[0]
        return None

    # Extract first label from list of labels
    df_train['label'] = df_train['labels'].apply(get_first_label)
    df_valid['label'] = df_valid['labels'].apply(get_first_label)

    # Select feature column and labels
    X_train = df_train[['text']]
    y_train = df_train['label']

    X_valid = df_valid[['text']]
    y_valid = df_valid['label']

    return X_train, y_train, X_valid, y_valid

@flow(name="mlflow-training", task_runner=SequentialTaskRunner())
def main():
    # Set MLflow tracking server URI and experiment name
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(f"goemotions-classification-{datetime.now().strftime('%Y-%m-%d')}")

    logger = get_run_logger()
    logger.info("Loading and processing GoEmotions dataset")
    X_train, y_train, X_valid, y_valid = process_data()
    logger.info(f"Train and Validation shapes: X_train={X_train.shape}, y_train={len(y_train)}, X_valid={X_valid.shape}, y_valid={len(y_valid)}")

    logger.info("Training models")
    run_models(X_train, y_train, X_valid, y_valid)


if __name__ == "__main__":
    main()
