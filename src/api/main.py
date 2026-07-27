"""Credit Risk prediction API.

Loads the fitted sklearn Pipeline (preprocessing + classifier bundled
together, see src/train.py) and serves risk probability predictions.
Feature order is driven by config.PipelineConfig.feature_columns so the
API can never silently drift out of sync with what the model was trained
on.
"""

from __future__ import annotations

import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import PredictionRequest, PredictionResponse
from src.config import DEFAULT_CONFIG

app = FastAPI(title="Credit Risk API")

try:
    _model = joblib.load(DEFAULT_CONFIG.model_artifact_path)
except FileNotFoundError:
    # Allows the app to start (e.g. for API doc generation or import-time
    # tests) even before a model has been trained; /predict will raise a
    # clear 503 instead of crashing at import time.
    _model = None


@app.get("/")
def home() -> dict:
    return {"message": "Credit Risk Model API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest) -> PredictionResponse:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python -m src.train` to train and save a model first.",
        )

    # Building a DataFrame (not a bare numpy array) with the exact training
    # column names/order avoids the original silent feature mismatch bug.
    features = pd.DataFrame(
        [[getattr(data, col) for col in DEFAULT_CONFIG.feature_columns]],
        columns=list(DEFAULT_CONFIG.feature_columns),
    )

    probability = float(_model.predict_proba(features)[0][1])
    is_high_risk = int(probability > DEFAULT_CONFIG.decision_threshold)

    return PredictionResponse(
        risk_probability=probability, is_high_risk=is_high_risk
    )
