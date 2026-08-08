"""
Data pipeline: load raw data, clean, validate, engineer features.

Run:
    python src/data_pipeline.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

RAW_PATH = Path(__file__).parent.parent / "data" / "customers_raw.csv"
PROCESSED_PATH = Path(__file__).parent.parent / "data" / "customers_processed.csv"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data/generate_data.py` first."
        )
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Impute missing monthly_charges with median
    df["monthly_charges"] = df["monthly_charges"].fillna(df["monthly_charges"].median())
    # Guard against negative/invalid tenure
    df = df[df["tenure_months"] >= 0]
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["avg_charge_per_tenure"] = df["monthly_charges"] / (df["tenure_months"] + 1)
    df["is_month_to_month"] = (df["contract_type"] == "Month-to-month").astype(int)
    df["high_support_usage"] = (df["support_calls"] >= 3).astype(int)
    df["tenure_bucket"] = pd.cut(
        df["tenure_months"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6mo", "6-12mo", "1-2yr", "2-4yr", "4-6yr"],
    )
    return df


def validate(df: pd.DataFrame) -> None:
    assert df["monthly_charges"].isna().sum() == 0, "Unhandled nulls in monthly_charges"
    assert df["churn"].isin([0, 1]).all(), "Invalid churn label"
    assert (df["tenure_months"] >= 0).all(), "Negative tenure found"


def run_pipeline(raw_path: Path = RAW_PATH, out_path: Path = PROCESSED_PATH) -> pd.DataFrame:
    df = load_raw(raw_path)
    df = clean(df)
    df = engineer_features(df)
    validate(df)
    df.to_csv(out_path, index=False)
    print(f"Processed {len(df)} rows -> {out_path}")
    return df


if __name__ == "__main__":
    run_pipeline()
