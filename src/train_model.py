"""
Train a churn-prediction model and save the artifact + metrics.

Run:
    python src/train_model.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_pipeline import run_pipeline, PROCESSED_PATH

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "support_calls",
    "avg_charge_per_tenure",
    "is_month_to_month",
    "high_support_usage",
]
CATEGORICAL_FEATURES = [
    "contract_type",
    "payment_method",
    "internet_service",
    "partner",
    "dependents",
]
TARGET = "churn"


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = GradientBoostingClassifier(random_state=42)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def train():
    if not PROCESSED_PATH.exists():
        df = run_pipeline()
    else:
        df = pd.read_csv(PROCESSED_PATH)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Model saved -> {MODEL_PATH}")
    return pipeline, metrics


if __name__ == "__main__":
    train()
