import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

import pandas as pd
from data_pipeline import clean, engineer_features, validate


def _sample_df():
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "tenure_months": [5, 20],
            "monthly_charges": [50.0, None],
            "contract_type": ["Month-to-month", "Two year"],
            "support_calls": [1, 4],
            "payment_method": ["Credit card", "Mailed check"],
            "internet_service": ["DSL", "Fiber optic"],
            "partner": ["Yes", "No"],
            "dependents": ["No", "Yes"],
            "churn": [0, 1],
        }
    )


def test_clean_fills_missing_charges():
    df = clean(_sample_df())
    assert df["monthly_charges"].isna().sum() == 0


def test_engineer_features_adds_columns():
    df = engineer_features(clean(_sample_df()))
    for col in ["avg_charge_per_tenure", "is_month_to_month", "high_support_usage", "tenure_bucket"]:
        assert col in df.columns


def test_validate_passes_on_clean_data():
    df = engineer_features(clean(_sample_df()))
    validate(df)  # should not raise
