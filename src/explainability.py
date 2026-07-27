"""Model explainability utilities built on SHAP.

Answers the three questions the business needs from an explainable credit
model:
  1. Which features matter most globally? -> global_feature_importance / summary_plot
  2. Why did the model make this specific prediction? -> explain_instance / force_plot
  3. Are there concerning patterns? -> dependence data returned alongside SHAP values
    for the caller (dashboard) to inspect per-feature trends.

Works with any fitted sklearn.pipeline.Pipeline whose final step is a
classifier and whose earlier steps are invertible/transparent (a
StandardScaler, as used in src/train.py). SHAP explanations are computed on
the *scaled* feature space produced by the pipeline's preprocessing steps,
then mapped back to the original feature names for readability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


@dataclass
class ExplanationResult:
    """SHAP explanation output for a batch of instances."""

    shap_values: np.ndarray          # shape (n_samples, n_features)
    base_value: float                # expected value (model output with no features)
    feature_names: List[str]
    raw_features: pd.DataFrame       # original (unscaled) feature values, for display


def _split_pipeline(pipeline: Pipeline):
    """Split a fitted Pipeline into its preprocessing steps and final estimator."""
    *preprocess_steps, (_, estimator) = pipeline.steps
    return preprocess_steps, estimator


def _transform_features(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Apply every step of the pipeline except the final estimator."""
    Xt = X
    for _, step in pipeline.steps[:-1]:
        Xt = step.transform(Xt)
    return Xt


def build_explainer(pipeline: Pipeline, background: pd.DataFrame) -> shap.Explainer:
    """Build a SHAP explainer for the classifier step of a fitted pipeline.

    ``background`` should be a representative sample of training data (raw,
    unscaled feature values) used to estimate the model's baseline output.
    """
    _, estimator = _split_pipeline(pipeline)
    background_transformed = _transform_features(pipeline, background)
    return shap.TreeExplainer(estimator) if _is_tree_model(estimator) else shap.Explainer(
        estimator.predict_proba, background_transformed
    )


def _is_tree_model(estimator) -> bool:
    return estimator.__class__.__name__ in {
        "RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier", "LGBMClassifier"
    }


def explain(
    pipeline: Pipeline,
    explainer: shap.Explainer,
    X: pd.DataFrame,
    feature_names: List[str],
) -> ExplanationResult:
    """Compute SHAP values for one or more instances."""
    X_transformed = _transform_features(pipeline, X)

    if _is_tree_model_explainer(explainer):
        raw_shap = explainer.shap_values(X_transformed)
    else:
        raw_shap = explainer(X_transformed)

    if hasattr(raw_shap, "values"):
        values = raw_shap.values
        if values.ndim == 3:
            values = values[:, :, 1]  # positive class
        base_value = float(np.mean(raw_shap.base_values))
    elif isinstance(raw_shap, list):
        # Older SHAP: list of per-class arrays [class0, class1]
        values = raw_shap[1]
        base_value = float(explainer.expected_value[1]) if isinstance(
            explainer.expected_value, (list, np.ndarray)
        ) else float(explainer.expected_value)
    else:
        # Current SHAP TreeExplainer: ndarray shaped (n_samples, n_features, n_classes)
        values = raw_shap[:, :, 1] if raw_shap.ndim == 3 else raw_shap
        base_value = float(explainer.expected_value[1]) if isinstance(
            explainer.expected_value, (list, np.ndarray)
        ) else float(explainer.expected_value)

    return ExplanationResult(
        shap_values=np.asarray(values),
        base_value=base_value,
        feature_names=feature_names,
        raw_features=X.reset_index(drop=True),
    )


def _is_tree_model_explainer(explainer) -> bool:
    return isinstance(explainer, shap.TreeExplainer)


def global_feature_importance(result: ExplanationResult) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, sorted descending (answers Q1)."""
    mean_abs = np.abs(result.shap_values).mean(axis=0)
    importance = pd.DataFrame({
        "feature": result.feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return importance


def explain_instance(result: ExplanationResult, index: int) -> pd.DataFrame:
    """Per-feature SHAP contributions for a single instance (answers Q2).

    Positive shap_value pushes the prediction toward high-risk; negative
    pushes toward low-risk.
    """
    row = pd.DataFrame({
        "feature": result.feature_names,
        "feature_value": result.raw_features.iloc[index][result.feature_names].values,
        "shap_value": result.shap_values[index],
    }).sort_values("shap_value", key=np.abs, ascending=False).reset_index(drop=True)
    return row
