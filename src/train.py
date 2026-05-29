import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.data_processing import (
    preprocess_data,
    build_rfm_target
)


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, preds)
    }


def main():
    path = "data/raw/data.csv"

    df = preprocess_data(path)
    rfm = build_rfm_target(df)

    data = rfm.copy()

    X = pd.get_dummies(data[["CustomerId"]])
    y = data["is_high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # =====================
    # LOGISTIC REGRESSION
    # =====================
    log_model = LogisticRegression(max_iter=1000)

    log_model.fit(X_train, y_train)

    log_results = evaluate(
        log_model,
        X_test,
        y_test
    )

    print(
        "Logistic Regression:",
        log_results
    )

    # =====================
    # RANDOM FOREST
    # =====================
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    rf_results = evaluate(
        rf_model,
        X_test,
        y_test
    )

    print(
        "Random Forest:",
        rf_results
    )

    # =====================
    # SAVE MODEL
    # =====================
    joblib.dump(rf_model, "model.pkl")

    print("Model saved as model.pkl")


if __name__ == "__main__":
    main()
