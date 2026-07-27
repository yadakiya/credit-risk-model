"""Train, evaluate, and track credit risk classification models.

Fixes the original bug where the model was trained on one-hot encoded
CustomerId values while the API served predictions from aggregate
transaction features -- a mismatch that made every /predict call fail.
Both training and serving now share ``config.PipelineConfig.feature_columns``
as the single source of truth for what the model expects.

Each candidate model is a single fitted sklearn.pipeline.Pipeline
(StandardScaler -> classifier), so the exact same object handles scaling
and prediction at inference time -- no separate scaler to keep in sync.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_CONFIG, PipelineConfig
from src.data_processing import build_training_dataset, preprocess_data

RAW_DATA_PATH = "data/raw/data.csv"


def evaluate(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """Compute standard binary classification metrics for a fitted model."""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs),
    }


def build_candidate_models(config: PipelineConfig) -> Dict[str, Pipeline]:
    """Define candidate model pipelines (preprocessing + estimator, single object)."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000, random_state=config.random_state
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100, random_state=config.random_state
                    ),
                ),
            ]
        ),
    }


def train_and_track(config: PipelineConfig = DEFAULT_CONFIG) -> str:
    """Train all candidate models, log each run to MLflow, and register the best.

    Returns the name of the best-performing model (by ROC-AUC).
    """
    df = preprocess_data(RAW_DATA_PATH)
    dataset = build_training_dataset(df, config=config)

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(processed_dir / "training_dataset.csv", index=False)

    X = dataset[list(config.feature_columns)]
    y = dataset["is_high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    mlflow.set_experiment(config.mlflow_experiment_name)

    best_name = None
    best_score = -1.0
    best_pipeline = None

    for name, pipeline in build_candidate_models(config).items():
        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)
            metrics = evaluate(pipeline, X_test, y_test)

            mlflow.log_params(
                {
                    "model_type": name,
                    "random_state": config.random_state,
                    "test_size": config.test_size,
                    "n_clusters": config.n_clusters,
                    "feature_columns": ",".join(config.feature_columns),
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            print(f"{name}: {metrics}")

            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_name = name
                best_pipeline = pipeline

    # Register the best model in the MLflow Model Registry.
    with mlflow.start_run(run_name=f"{best_name}_registered"):
        mlflow.sklearn.log_model(
            best_pipeline,
            artifact_path="model",
            registered_model_name=config.mlflow_registered_model_name,
        )

    # Also persist to a plain joblib artifact for lightweight local serving
    # (used as a fallback by the API when no MLflow tracking server is
    # configured -- e.g. in the Docker container).
    joblib.dump(best_pipeline, config.model_artifact_path)

    print(
        f"Best model: {best_name} (ROC-AUC={best_score:.4f}), "
        f"saved to {config.model_artifact_path}"
    )
    return best_name


def main() -> None:
    train_and_track()


if __name__ == "__main__":
    main()
