# EDA Plan

Convert this into `eda.ipynb` locally and explore:

1. Churn rate overall and by `contract_type`, `internet_service`, `payment_method`
2. Distribution of `tenure_months` and `monthly_charges`, split by churn
3. Correlation between `support_calls` and churn
4. Missingness in `monthly_charges` — pattern or random?
5. Feature importance from the trained model (`models/metrics.json`, SHAP output)

```python
import pandas as pd
df = pd.read_csv("../data/customers_processed.csv")
df.groupby("contract_type")["churn"].mean()
```
