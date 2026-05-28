import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    return df


def convert_datetime(df):
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    df["hour"] = df["TransactionStartTime"].dt.hour
    df["day"] = df["TransactionStartTime"].dt.day
    df["month"] = df["TransactionStartTime"].dt.month

    return df


def aggregate_features(df):
    customer_df = df.groupby("CustomerId").agg(
        total_amount=("Amount", "sum"),
        avg_amount=("Amount", "mean"),
        transaction_count=("TransactionId", "count"),
        std_amount=("Amount", "std"),
        max_amount=("Amount", "max"),
        min_amount=("Amount", "min")
    ).reset_index()

    return customer_df


def handle_missing(df):
    return df.fillna(0)


def preprocess_data(path):
    df = load_data(path)

    df = convert_datetime(df)

    customer_df = aggregate_features(df)

    customer_df = handle_missing(customer_df)

    return customer_df
