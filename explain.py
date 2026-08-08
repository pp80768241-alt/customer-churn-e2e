"""
Model explainability using SHAP.

Given a trained pipeline and a single row of features, returns the
top contributing features for that prediction.
"""

from pathlib import Path
import joblib
import pandas as pd
import shap

MODEL_PATH = Path(__file__).parent.parent / "models" / "churn_model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python src/train_model.py` first.")
    return joblib.load(MODEL_PATH)


def explain_prediction(pipeline, row: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """
    Returns the top_n features driving the prediction for a single-row DataFrame.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    transformed = preprocessor.transform(row)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed)

    contributions = list(zip(feature_names, shap_values[0]))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    return [
        {"feature": name, "impact": round(float(val), 4)}
        for name, val in contributions[:top_n]
    ]


if __name__ == "__main__":
    import pandas as pd

    pipeline = load_model()
    sample = pd.DataFrame(
        [
            {
                "tenure_months": 3,
                "monthly_charges": 95.0,
                "support_calls": 4,
                "avg_charge_per_tenure": 95.0 / 4,
                "is_month_to_month": 1,
                "high_support_usage": 1,
                "contract_type": "Month-to-month",
                "payment_method": "Electronic check",
                "internet_service": "Fiber optic",
                "partner": "No",
                "dependents": "No",
            }
        ]
    )
    print(explain_prediction(pipeline, sample))
