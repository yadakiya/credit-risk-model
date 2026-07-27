"""Interactive credit risk dashboard for Bati Bank's risk team.

Run with:
    streamlit run dashboard/app.py

Two views:
  - "Score a customer": enter a customer's aggregate transaction features,
    get a risk probability, and see exactly which features drove that
    specific decision (SHAP waterfall) -- answers "why did the model make
    this prediction?" for a non-technical reviewer.
  - "Portfolio overview": if a trained model + training data are available,
    shows the global feature-importance ranking and the risk distribution
    across the whole customer base -- answers "which features matter most
    globally?" and "are there concerning patterns?"
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `streamlit run dashboard/app.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import DEFAULT_CONFIG  # noqa: E402
from src.explainability import (  # noqa: E402
    build_explainer,
    explain,
    explain_instance,
    global_feature_importance,
)

st.set_page_config(
    page_title="Bati Bank — Credit Risk Dashboard", layout="wide"
)

FEATURES = list(DEFAULT_CONFIG.feature_columns)
FEATURE_LABELS = {
    "total_amount": "Total transaction amount",
    "avg_amount": "Average transaction amount",
    "transaction_count": "Number of transactions",
    "std_amount": "Std. deviation of transaction amount",
}


@st.cache_resource
def load_model():
    try:
        return joblib.load(DEFAULT_CONFIG.model_artifact_path)
    except FileNotFoundError:
        return None


@st.cache_data
def load_training_dataset():
    """Optional: portfolio-level context if a processed dataset is present."""
    path = Path("data/processed/training_dataset.csv")
    if path.exists():
        return pd.read_csv(path)
    return None


def render_header():
    st.title("💳 Bati Bank — Credit Risk Dashboard")
    st.caption(
        "Buy-now-pay-later risk scoring for the eCommerce lending partnership. "
        "Business problem: new customers have no repayment history, so risk is "
        "estimated from their transaction behavior."
    )


def render_scoring_tab(model, background: pd.DataFrame):
    st.subheader("Score a customer")
    st.write(
        "Enter a customer's transaction summary to get a risk probability and see why."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        total_amount = st.number_input(
            "Total transaction amount",
            min_value=0.0,
            value=15000.0,
            step=100.0,
        )
        avg_amount = st.number_input(
            "Average transaction amount", min_value=0.0, value=750.0, step=10.0
        )
        transaction_count = st.number_input(
            "Number of transactions", min_value=0, value=20, step=1
        )
        std_amount = st.number_input(
            "Std. deviation of transaction amount",
            min_value=0.0,
            value=320.0,
            step=10.0,
        )
        run_button = st.button("Score customer", type="primary")

    if not run_button:
        st.info("Enter values and click **Score customer**.")
        return

    if model is None:
        st.error(
            "No trained model found. Run `python -m src.train` first to produce model.pkl."
        )
        return

    input_row = pd.DataFrame(
        [
            {
                "total_amount": total_amount,
                "avg_amount": avg_amount,
                "transaction_count": transaction_count,
                "std_amount": std_amount,
            }
        ]
    )[FEATURES]

    probability = float(model.predict_proba(input_row)[0][1])
    is_high_risk = probability > DEFAULT_CONFIG.decision_threshold

    with col2:
        risk_color = "🔴" if is_high_risk else "🟢"
        st.metric("Risk probability", f"{probability:.1%}")
        st.markdown(
            f"### {risk_color} {'High risk' if is_high_risk else 'Low risk'}"
        )
        st.progress(min(max(probability, 0.0), 1.0))

        st.markdown("#### Why this prediction?")
        explainer = build_explainer(model, background=background)
        result = explain(model, explainer, input_row, feature_names=FEATURES)
        contributions = explain_instance(result, index=0)
        contributions["label"] = contributions["feature"].map(FEATURE_LABELS)

        fig, ax = plt.subplots(figsize=(6, 3))
        colors = [
            "#d62728" if v > 0 else "#2ca02c"
            for v in contributions["shap_value"]
        ]
        ax.barh(
            contributions["label"], contributions["shap_value"], color=colors
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(
            "Contribution to risk score (red = increases risk, green = decreases risk)"
        )
        ax.invert_yaxis()
        st.pyplot(fig)


def render_portfolio_tab(model, background: pd.DataFrame):
    st.subheader("Portfolio overview")

    if model is None:
        st.warning("No trained model found. Run `python -m src.train` first.")
        return

    st.markdown("#### Which features matter most globally?")
    explainer = build_explainer(model, background=background)
    sample = background.sample(min(len(background), 100), random_state=42)
    result = explain(model, explainer, sample, feature_names=FEATURES)
    importance = global_feature_importance(result)
    importance["label"] = importance["feature"].map(FEATURE_LABELS)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(importance["label"], importance["mean_abs_shap"], color="#1f77b4")
    ax.set_xlabel("Mean |SHAP value| (average impact on risk score)")
    ax.invert_yaxis()
    st.pyplot(fig)

    dataset = load_training_dataset()
    if dataset is not None and "is_high_risk" in dataset.columns:
        st.markdown("#### Risk distribution across the customer base")
        risk_counts = (
            dataset["is_high_risk"]
            .value_counts()
            .rename({0: "Low risk", 1: "High risk"})
        )
        st.bar_chart(risk_counts)
        pct_high_risk = dataset["is_high_risk"].mean() * 100
        st.caption(
            f"{pct_high_risk:.1f}% of customers in the training set "
            "fall in the high-risk segment."
        )
    else:
        st.caption(
            "No processed training dataset found at data/processed/training_dataset.csv "
            "-- showing feature importance only. Save the merged dataset there for full "
            "portfolio-level views."
        )


def main():
    render_header()

    model = load_model()
    dataset = load_training_dataset()
    # Background sample for SHAP: prefer real training data, fall back to a
    # small synthetic sample so the dashboard is still usable pre-training.
    if dataset is not None:
        background = dataset[FEATURES].sample(
            min(len(dataset), 50), random_state=42
        )
    else:
        background = pd.DataFrame(
            [
                {
                    "total_amount": 5000.0,
                    "avg_amount": 300.0,
                    "transaction_count": 15,
                    "std_amount": 150.0,
                }
            ]
            * 10
        )

    tab1, tab2 = st.tabs(["Score a customer", "Portfolio overview"])
    with tab1:
        render_scoring_tab(model, background)
    with tab2:
        render_portfolio_tab(model, background)


if __name__ == "__main__":
    main()
