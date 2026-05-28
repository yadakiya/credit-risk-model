import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# =========================
# BASIC DATA LOADING + CLEANING
# =========================

def load_data(path):
    return pd.read_csv(path)


def convert_datetime(df):
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    return df


def handle_missing(df):
    return df.fillna(0)


def preprocess_data(path):
    df = load_data(path)
    df = convert_datetime(df)
    df = handle_missing(df)
    return df


# =========================
# TASK 3: CUSTOMER FEATURES
# =========================

def aggregate_customer_features(df):
    customer_df = df.groupby("CustomerId").agg(
        total_amount=("Amount", "sum"),
        avg_amount=("Amount", "mean"),
        transaction_count=("TransactionId", "count")
    ).reset_index()

    return customer_df


# =========================
# TASK 4: RFM FEATURES
# =========================

def create_rfm(df):
    snapshot_date = df["TransactionStartTime"].max()

    rfm = df.groupby("CustomerId").agg(
        Recency=(
            "TransactionStartTime",
            lambda x: (
                snapshot_date -
                x.max()).days),
        Frequency=(
            "TransactionId",
            "count"),
        Monetary=(
            "Amount",
            "sum")).reset_index()

    return rfm


def cluster_customers(rfm):
    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    return rfm


def label_risk(rfm):
    summary = rfm.groupby("cluster")[
        ["Recency", "Frequency", "Monetary"]].mean()

    high_risk_cluster = summary["Recency"].idxmax()

    rfm["is_high_risk"] = rfm["cluster"].apply(
        lambda x: 1 if x == high_risk_cluster else 0
    )

    return rfm


def build_rfm_target(df):
    rfm = create_rfm(df)
    rfm = cluster_customers(rfm)
    rfm = label_risk(rfm)

    return rfm[["CustomerId", "is_high_risk"]]
