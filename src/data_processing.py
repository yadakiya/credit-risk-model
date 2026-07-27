"""Data loading, feature engineering, and RFM-based proxy target creation.

This module transforms raw Xente transaction records into a model-ready
customer-level dataset. The critical invariant enforced here is that the
columns produced by ``aggregate_customer_features`` are exactly the columns
the trained model consumes at inference time (see ``config.PipelineConfig
.feature_columns``) -- an earlier version of this pipeline trained on a
different, unrelated set of columns (one-hot encoded CustomerId) than what
the API sent at prediction time, which meant every /predict call raised a
feature-mismatch error. ``build_training_dataset`` below is the single
source of truth that ties feature engineering and target labeling together
and is covered by regression tests in tests/test_data_processing.py.
"""
from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_CONFIG, PipelineConfig


def load_data(path: str) -> pd.DataFrame:
    """Load raw transaction data from a CSV file."""
    return pd.read_csv(path)


def preprocess_data(path: str) -> pd.DataFrame:
    """Load and lightly clean raw transaction data.

    Parses transaction timestamps and fills missing values. Does not
    aggregate or engineer features -- see ``aggregate_customer_features``
    and ``create_rfm`` for that.
    """
    df = load_data(path)
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    df = df.fillna(0)
    return df


# =========================
# TASK 3: AGGREGATE FEATURES
# =========================

def aggregate_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build customer-level aggregate features from transaction records.

    Returns one row per CustomerId with total/average/count/std of
    transaction amounts. These are the exact columns the API accepts at
    inference time.
    """
    agg = df.groupby("CustomerId").agg(
        total_amount=("Amount", "sum"),
        avg_amount=("Amount", "mean"),
        transaction_count=("TransactionId", "count"),
        std_amount=("Amount", "std"),
    ).reset_index()

    # A customer with a single transaction has an undefined std; treat as 0
    # rather than leaking NaNs into the model.
    agg["std_amount"] = agg["std_amount"].fillna(0)
    return agg


# =========================
# TASK 4: RFM PROXY TARGET
# =========================

def create_rfm(
    df: pd.DataFrame,
    snapshot_offset_days: int = DEFAULT_CONFIG.snapshot_offset_days,
) -> pd.DataFrame:
    """Compute Recency, Frequency, and Monetary value per customer."""
    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(
        days=snapshot_offset_days
    )

    rfm = df.groupby("CustomerId").agg(
        Recency=(
            "TransactionStartTime",
            lambda x: (snapshot_date - x.max()).days,
        ),
        Frequency=("TransactionId", "count"),
        Monetary=("Amount", "sum"),
    ).reset_index()

    return rfm


def cluster_customers(
    rfm: pd.DataFrame,
    n_clusters: int = DEFAULT_CONFIG.n_clusters,
    random_state: int = DEFAULT_CONFIG.random_state,
) -> pd.DataFrame:
    """Segment customers into behavioral clusters using scaled RFM values."""
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm = rfm.copy()
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)
    return rfm


def label_risk(rfm: pd.DataFrame) -> pd.DataFrame:
    """Label the least-engaged cluster (highest Recency) as high-risk."""
    cluster_summary = rfm.groupby("cluster")[
        ["Recency", "Frequency", "Monetary"]
    ].mean()

    high_risk_cluster = cluster_summary["Recency"].idxmax()

    rfm = rfm.copy()
    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)
    return rfm


def build_rfm_target(
    df: pd.DataFrame, config: PipelineConfig = DEFAULT_CONFIG
) -> pd.DataFrame:
    """Run the full RFM -> clustering -> labeling flow, returning the proxy target."""
    rfm = create_rfm(df, snapshot_offset_days=config.snapshot_offset_days)
    rfm = cluster_customers(
        rfm, n_clusters=config.n_clusters, random_state=config.random_state
    )
    rfm = label_risk(rfm)
    return rfm[["CustomerId", "is_high_risk"]]


# =========================
# MODEL-READY DATASET
# =========================

def build_training_dataset(
    df: pd.DataFrame, config: PipelineConfig = DEFAULT_CONFIG
) -> pd.DataFrame:
    """Merge aggregate features with the RFM proxy target on CustomerId.

    This is the model-ready dataset: one row per customer, with the feature
    columns the model is trained on (config.feature_columns) and the
    ``is_high_risk`` label. Training and serving both depend on
    config.feature_columns as their single source of truth, so the
    columns cannot silently drift apart.
    """
    features = aggregate_customer_features(df)
    target = build_rfm_target(df, config=config)
    return features.merge(target, on="CustomerId", how="inner")
