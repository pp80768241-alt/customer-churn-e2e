"""
FastAPI service for churn prediction.

Run:
    uvicorn api.main:app --reload
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from explain import explain_prediction  # noqa: E402
from train_model import MODEL_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES  # noqa: E402

app = FastAPI(
    title="Customer Churn Prediction API",
    description="End-to-end ML system: predicts customer churn and explains why.",
    version="1.0.0",
)

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not trained yet. Run `python src/train_model.py`.",
            )
        _model = joblib.load(MODEL_PATH)
    return _model


class CustomerFeatures(BaseModel):
    tenure_months: int = Field(..., ge=0, le=100, example=12)
    monthly_charges: float = Field(..., ge=0, example=65.5)
    support_calls: int = Field(..., ge=0, example=3)
    contract_type: str = Field(..., example="Month-to-month")
    payment_method: str = Field(default="Electronic check", example="Electronic check")
    internet_service: str = Field(default="Fiber optic", example="Fiber optic")
    partner: str = Field(default="No", example="No")
    dependents: str = Field(default="No", example="No")


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool


def _to_row(features: CustomerFeatures) -> pd.DataFrame:
    data = features.dict()
    data["avg_charge_per_tenure"] = data["monthly_charges"] / (data["tenure_months"] + 1)
    data["is_month_to_month"] = int(data["contract_type"] == "Month-to-month")
    data["high_support_usage"] = int(data["support_calls"] >= 3)
    return pd.DataFrame([data])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    model = get_model()
    row = _to_row(features)
    proba = float(model.predict_proba(row)[0, 1])
    return PredictionResponse(churn_probability=round(proba, 4), churn_prediction=proba >= 0.5)


@app.post("/explain")
def explain(features: CustomerFeatures):
    model = get_model()
    row = _to_row(features)
    top_features = explain_prediction(model, row)
    return {"top_features": top_features}
