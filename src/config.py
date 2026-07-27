"""Centralized, typed configuration for the credit risk pipeline.

Replaces magic numbers that were previously hardcoded inline across
data_processing.py and train.py.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class PipelineConfig:
    """Reproducibility and modeling constants for the credit risk pipeline."""

    random_state: int = 42
    n_clusters: int = 3
    test_size: float = 0.2
    snapshot_offset_days: int = 1
    decision_threshold: float = 0.5

    # The feature columns the model is trained on AND the columns the API
    # must send at inference time. Defining this once in one place is what
    # prevents the training/serving feature set from silently drifting
    # apart (the root cause of the original train/predict mismatch bug).
    feature_columns: Tuple[str, ...] = field(
        default_factory=lambda: (
            "total_amount",
            "avg_amount",
            "transaction_count",
            "std_amount",
        )
    )

    model_artifact_path: str = "model.pkl"
    mlflow_experiment_name: str = "credit-risk-model"
    mlflow_registered_model_name: str = "credit-risk-classifier"


DEFAULT_CONFIG = PipelineConfig()
