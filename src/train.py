import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.data_processing import preprocess_data, build_rfm_target


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred)
    }


def main():
    path = "data/raw/data.csv"

    # ======================
    # LOAD DATA
    # ======================
    df = preprocess_data(path)

    rfm = build_rfm_target(df)

    # Merge target back if needed (simple demo)
    data = rfm.copy()

    X = data[["CustomerId"]]  # placeholder (we'll improve in Task 6)
    y = data["is_high_risk"]

    # Convert CustomerId to numeric
    X = pd.get_dummies(X)

    # ======================
    # TRAIN TEST SPLIT
    # ======================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ======================
    # MODEL 1: LOGISTIC REGRESSION
    # ======================
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)

    log_metrics = evaluate_model(log_model, X_test, y_test)

    print("\nLogistic Regression Results:")
    print(log_metrics)

    # ======================
    # MODEL 2: RANDOM FOREST
    # ======================
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    rf_metrics = evaluate_model(rf_model, X_test, y_test)

    print("\nRandom Forest Results:")
    print(rf_metrics)


if __name__ == "__main__":
    main()
