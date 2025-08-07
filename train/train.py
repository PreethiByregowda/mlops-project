import pandas as pd
import re
import json
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
import ast

# ---------- Helper: Clean Label Strings ----------

def clean_label_string(label_str):
    try:
        if pd.isna(label_str):
            return []
        # Replace whitespace with commas (e.g., [ 8 20] → [8,20])
        cleaned = re.sub(r'\s+', ',', label_str.strip())
        # Evaluate string safely (e.g., '[1,2]' → [1, 2])
        return ast.literal_eval(cleaned)
    except Exception:
        return []

# ---------- Load and Clean Data ----------
train_df = pd.read_csv("data/train.csv")
valid_df = pd.read_csv("data/valid.csv")

train_df["labels"] = train_df["labels"].apply(clean_label_string)
valid_df["labels"] = valid_df["labels"].apply(clean_label_string)

X_train = train_df["text"]
X_valid = valid_df["text"]

mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_df["labels"])
y_valid = mlb.transform(valid_df["labels"])

# ---------- ML Pipeline ----------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", OneVsRestClassifier(
        LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    ))
])

# ---------- MLflow Setup ----------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("goemotions-classification-2025-08-06")

with mlflow.start_run():
    mlflow.sklearn.autolog(log_models=True)

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_valid)

    micro_f1 = f1_score(y_valid, preds, average="micro")
    macro_f1 = f1_score(y_valid, preds, average="macro")
    
    mlflow.log_metric("micro_f1", micro_f1)
    mlflow.log_metric("macro_f1", macro_f1)

    print(f"✅ Micro F1: {micro_f1:.4f}")
    print(f"✅ Macro F1: {macro_f1:.4f}")