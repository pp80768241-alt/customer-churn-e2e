# 📊 Customer Churn Prediction — End-to-End Analytics/ML System

An end-to-end analytics/ML project that takes raw customer data all the way to a deployed,
containerized prediction API — with CI/CD, monitoring hooks, and model explainability baked in.

This isn't "I know pandas" or "I know Flask" in isolation. It's a demonstration of:

**Build → Integrate → Deploy → Scale → Explain a real system.**

---

## 🧱 Architecture

```
Raw Data → Data Pipeline → Feature Engineering → Model Training (+ MLflow tracking)
                                                        │
                                                        ▼
                                            Serialized Model (models/)
                                                        │
                                                        ▼
                                    FastAPI Prediction Service (api/)
                                                        │
                                    ┌───────────────────┼───────────────────┐
                                    ▼                   ▼                   ▼
                              Docker Image      GitHub Actions CI     SHAP Explainability
                              (scalable,             (tests +           endpoint
                              deployable)            lint on push)
```

## 📁 Repository Structure

```
.
├── data/
│   └── generate_data.py       # Generates a realistic synthetic churn dataset
├── src/
│   ├── data_pipeline.py       # Cleaning, validation, feature engineering
│   ├── train_model.py         # Trains + evaluates model, saves artifact
│   └── explain.py             # SHAP-based model explainability
├── api/
│   └── main.py                # FastAPI app: /predict, /explain, /health
├── tests/
│   ├── test_pipeline.py
│   └── test_api.py
├── notebooks/
│   └── eda.md                 # EDA notes/plan (convert to .ipynb locally)
├── models/                    # Trained model artifacts land here (gitignored)
├── .github/workflows/ci.yml   # Lint + test on every push/PR
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .gitignore
```

## 🚀 Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/<your-username>/customer-churn-e2e.git
cd customer-churn-e2e
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Generate data
python data/generate_data.py

# 3. Run the pipeline + train the model
python src/data_pipeline.py
python src/train_model.py

# 4. Serve the API
uvicorn api.main:app --reload

# 5. Try it
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure_months": 12, "monthly_charges": 65.5, "contract_type": "Month-to-month", "support_calls": 3}'
```

## 🐳 Run with Docker

```bash
docker compose up --build
```

This builds the image, trains the model on container startup if no artifact exists, and
exposes the API on `http://localhost:8000`.

## 🧪 Tests & CI

```bash
pytest tests/ -v
```

GitHub Actions (`.github/workflows/ci.yml`) runs linting and the test suite on every push
and pull request.

## 🔍 Explainability

`GET /explain/{customer_id}` returns SHAP values showing which features pushed a given
prediction toward "churn" vs. "retain" — because a model recruiters can't interrogate
isn't a system they'll trust.

## 📈 What This Project Demonstrates

| Skill | Where |
|---|---|
| Data engineering / pipelines | `src/data_pipeline.py` |
| ML modeling & evaluation | `src/train_model.py` |
| API design | `api/main.py` |
| Containerization | `Dockerfile`, `docker-compose.yml` |
| CI/CD | `.github/workflows/ci.yml` |
| Testing | `tests/` |
| Explainability / model trust | `src/explain.py` |

## 📄 License

MIT — see `LICENSE`.
