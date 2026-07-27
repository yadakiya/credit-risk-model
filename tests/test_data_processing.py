import pandas as pd
import pytest

from src.config import PipelineConfig
from src.data_processing import (
    aggregate_customer_features,
    build_rfm_target,
    build_training_dataset,
    create_rfm,
    cluster_customers,
    label_risk,
)

TEST_CONFIG = PipelineConfig(random_state=42, n_clusters=3)


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    """A tiny synthetic transaction set with 3 distinct customer behaviors:
    a frequent big spender, an occasional small spender, and a customer
    who transacted once a long time ago (should land in the high-risk
    cluster: low frequency, low monetary, high recency).
    """
    rows = []
    # Customer A: frequent, high value, recent -> low risk
    for i in range(10):
        rows.append(
            {
                "TransactionId": f"a{i}",
                "CustomerId": "C_A",
                "Amount": 1000 + i * 10,
                "TransactionStartTime": f"2026-07-{10 + i:02d}",
            }
        )
    # Customer B: occasional, moderate value, recent-ish -> low/medium risk
    for i in range(4):
        rows.append(
            {
                "TransactionId": f"b{i}",
                "CustomerId": "C_B",
                "Amount": 200 + i * 5,
                "TransactionStartTime": f"2026-07-{15 + i:02d}",
            }
        )
    # Customer C: single transaction, long ago, tiny value -> high risk
    rows.append(
        {
            "TransactionId": "c0",
            "CustomerId": "C_C",
            "Amount": 10,
            "TransactionStartTime": "2026-01-01",
        }
    )

    df = pd.DataFrame(rows)
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    return df


def test_aggregate_customer_features_returns_expected_columns(
    sample_transactions,
):
    result = aggregate_customer_features(sample_transactions)

    expected_columns = {
        "CustomerId",
        "total_amount",
        "avg_amount",
        "transaction_count",
        "std_amount",
    }
    assert expected_columns.issubset(set(result.columns))
    assert len(result) == 3  # 3 unique customers


def test_aggregate_customer_features_values_are_correct(sample_transactions):
    result = aggregate_customer_features(sample_transactions).set_index(
        "CustomerId"
    )

    assert result.loc["C_A", "transaction_count"] == 10
    assert result.loc["C_C", "transaction_count"] == 1
    # single transaction -> no variance
    assert result.loc["C_C", "std_amount"] == 0


def test_create_rfm_returns_one_row_per_customer(sample_transactions):
    rfm = create_rfm(
        sample_transactions,
        snapshot_offset_days=TEST_CONFIG.snapshot_offset_days,
    )

    assert len(rfm) == 3
    assert {"CustomerId", "Recency", "Frequency", "Monetary"}.issubset(
        set(rfm.columns)
    )


def test_cluster_customers_assigns_a_cluster_to_every_row(sample_transactions):
    rfm = create_rfm(sample_transactions)
    clustered = cluster_customers(rfm, n_clusters=3, random_state=42)

    assert "cluster" in clustered.columns
    assert clustered["cluster"].notna().all()
    assert clustered["cluster"].nunique() <= 3


def test_label_risk_flags_the_least_engaged_customer_as_high_risk(
    sample_transactions,
):
    rfm = create_rfm(sample_transactions)
    rfm = cluster_customers(rfm, n_clusters=3, random_state=42)
    labeled = label_risk(rfm)

    assert set(labeled["is_high_risk"].unique()).issubset({0, 1})
    # Customer C (single old transaction, low value) should be labeled high-risk.
    c_row = labeled[labeled["CustomerId"] == "C_C"]
    assert c_row["is_high_risk"].iloc[0] == 1


def test_build_rfm_target_returns_only_id_and_label(sample_transactions):
    target = build_rfm_target(sample_transactions, config=TEST_CONFIG)

    assert set(target.columns) == {"CustomerId", "is_high_risk"}
    assert len(target) == 3


def test_build_training_dataset_merges_features_and_target_on_customer_id(
    sample_transactions,
):
    """Regression test for the original bug: the model must be trained on
    aggregate transaction features (total_amount, avg_amount,
    transaction_count, std_amount), not on a proxy like one-hot encoded
    CustomerId. This test fails loudly if that invariant is ever broken.
    """
    dataset = build_training_dataset(sample_transactions, config=TEST_CONFIG)

    assert len(dataset) == 3
    for col in TEST_CONFIG.feature_columns:
        assert col in dataset.columns
    assert "is_high_risk" in dataset.columns
    assert "CustomerId" in dataset.columns
