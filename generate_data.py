"""
Generates a synthetic customer churn dataset so the whole project is
runnable end-to-end with zero external downloads.

Run:
    python data/generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_CUSTOMERS = 5000

def generate_churn_dataset(n=N_CUSTOMERS, seed=RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(0, 72, size=n)
    monthly_charges = np.round(rng.normal(65, 25, size=n).clip(15, 150), 2)
    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20]
    )
    support_calls = rng.poisson(1.5, size=n)
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n
    )
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.35, 0.45, 0.20])
    partner = rng.choice(["Yes", "No"], size=n)
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.3, 0.7])

    # Construct churn probability from a plausible latent function
    logit = (
        -1.5
        - 0.04 * tenure_months
        + 0.015 * monthly_charges
        + 0.35 * support_calls
        + np.where(contract_type == "Month-to-month", 1.1, 0)
        + np.where(contract_type == "One year", 0.2, 0)
        + np.where(internet_service == "Fiber optic", 0.4, 0)
        + np.where(payment_method == "Electronic check", 0.3, 0)
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob_churn)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(n)],
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "contract_type": contract_type,
            "support_calls": support_calls,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "partner": partner,
            "dependents": dependents,
            "churn": churn,
        }
    )

    # Inject a bit of realistic messiness for the pipeline to clean
    missing_idx = rng.choice(n, size=int(n * 0.02), replace=False)
    df.loc[missing_idx, "monthly_charges"] = np.nan

    return df


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    df = generate_churn_dataset()
    out_path = out_dir / "customers_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(f"Churn rate: {df['churn'].mean():.2%}")
