from fastapi import FastAPI
import numpy as np
import joblib

from src.api.pydantic_models import PredictionRequest, PredictionResponse

app = FastAPI(title="Credit Risk API")


# Load model (you will save it in Task 5)
model = joblib.load("model.pkl")


@app.get("/")
def home():
    return {"message": "Credit Risk Model API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):

    features = np.array([[
        data.total_amount,
        data.avg_amount,
        data.transaction_count
    ]])

    prob = model.predict_proba(features)[0][1]
    prediction = int(prob > 0.5)

    return PredictionResponse(
        risk_probability=prob,
        is_high_risk=prediction
    )
