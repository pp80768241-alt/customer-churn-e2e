FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate data, run pipeline, and train model at build time so the
# image ships ready to serve predictions.
RUN python data/generate_data.py \
    && python src/data_pipeline.py \
    && python src/train_model.py

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
