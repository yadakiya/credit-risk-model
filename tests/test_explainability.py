import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_CONFIG
from src.explainability import (
    build_explainer,
    explain,
    explain_instance,
    global_feature_importance,
)


@pytest.fixture
def synthetic_dataset():
    rng = np.random.RandomState(42)
    n = 100
    X = pd.DataFrame(
        {
            "total_amount": rng.uniform(100, 20000, n),
            "avg_amount": rng.uniform(50, 1000, n),
            "transaction_count": rng.randint(1, 50, n),
            "std_amount": rng.uniform(0, 500, n),
        }
    )
    y = ((X["transaction_count"] < 10) & (X["total_amount"] < 3000)).astype(
        int
    )
    return X, y


@pytest.mark.parametrize(
    "classifier",
    [
        RandomForestClassifier(n_estimators=30, random_state=42),
        LogisticRegression(random_state=42),
    ],
)
def test_explain_returns_one_shap_row_per_instance(
    synthetic_dataset, classifier
):
    X, y = synthetic_dataset
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("classifier", classifier)]
    ).fit(X, y)

    explainer = build_explainer(
        pipeline, background=X.sample(30, random_state=1)
    )
    sample = X.sample(10, random_state=2)
    result = explain(
        pipeline,
        explainer,
        sample,
        feature_names=list(DEFAULT_CONFIG.feature_columns),
    )

    assert result.shap_values.shape == (
        10,
        len(DEFAULT_CONFIG.feature_columns),
    )


def test_global_feature_importance_sums_to_ranked_features(synthetic_dataset):
    X, y = synthetic_dataset
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(n_estimators=30, random_state=42),
            ),
        ]
    ).fit(X, y)

    explainer = build_explainer(
        pipeline, background=X.sample(30, random_state=1)
    )
    result = explain(
        pipeline,
        explainer,
        X.sample(20, random_state=2),
        feature_names=list(DEFAULT_CONFIG.feature_columns),
    )
    importance = global_feature_importance(result)

    assert set(importance["feature"]) == set(DEFAULT_CONFIG.feature_columns)
    assert (
        importance["mean_abs_shap"].values[:-1]
        >= importance["mean_abs_shap"].values[1:]
    ).all()


def test_explain_instance_returns_one_row_per_feature(synthetic_dataset):
    X, y = synthetic_dataset
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(n_estimators=30, random_state=42),
            ),
        ]
    ).fit(X, y)

    explainer = build_explainer(
        pipeline, background=X.sample(30, random_state=1)
    )
    result = explain(
        pipeline,
        explainer,
        X.sample(5, random_state=2),
        feature_names=list(DEFAULT_CONFIG.feature_columns),
    )
    instance_explanation = explain_instance(result, index=0)

    assert len(instance_explanation) == len(DEFAULT_CONFIG.feature_columns)
    assert {"feature", "feature_value", "shap_value"}.issubset(
        instance_explanation.columns
    )
