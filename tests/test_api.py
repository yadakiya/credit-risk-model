"""Integration tests for the Credit Risk API.

Trains a tiny real pipeline in-memory and points the API's loaded model
at it, so these tests exercise the actual request -> feature-frame ->
predict_proba -> response path (the exact path that was broken by the
original feature mismatch bug) without requiring a full training run or
real data file.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_CONFIG


@pytest.fixture
def client(monkeypatch):
    import src.api.main as api_main

    rng = np.random.RandomState(42)
    X = pd.DataFrame(
        rng.rand(40, len(DEFAULT_CONFIG.feature_columns)) * 1000,
        columns=list(DEFAULT_CONFIG.feature_columns),
    )
    y = (X["transaction_count"] < 500).astype(int)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )
    pipeline.fit(X, y)

    monkeypatch.setattr(api_main, "_model", pipeline)
    return TestClient(api_main.app)


def test_home_endpoint_returns_ok(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_endpoint_returns_valid_response_shape(client):
    payload = {
        "total_amount": 15000.0,
        "avg_amount": 750.0,
        "transaction_count": 20,
        "std_amount": 320.5,
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["risk_probability"] <= 1.0
    assert body["is_high_risk"] in (0, 1)


def test_predict_endpoint_rejects_missing_required_field(client):
    payload = {
        "total_amount": 100.0,
        "avg_amount": 50.0,
    }  # missing transaction_count
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
