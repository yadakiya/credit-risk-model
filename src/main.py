from fastapi import FastAPI
import numpy as np
import joblib

from src.api.pydantic_models import PredictionRequest, PredictionResponse

app = FastAPI(title="Credit Risk API")

model = joblib.load("model.pkl")


@app.get("/")
def home():
    return {"message": "API running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):

    X = np.array([[data.total_amount, data.avg_amount, data.transaction_count]])

    prob = model.predict_proba(X)[0][1]
    pred = int(prob > 0.5)

    return PredictionResponse(
        risk_probability=prob,
        is_high_risk=pred
    )
