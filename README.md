# 💳 Credit Risk Modeling Project

## 📌 Overview
This project is an end-to-end machine learning system for credit risk prediction. It includes data preprocessing, feature engineering (RFM), clustering, model training, evaluation, and deployment using FastAPI. A CI/CD pipeline is also implemented using GitHub Actions.

---

## 🎯 Objectives
- Clean and preprocess raw transaction data
- Build customer-level features
- Create RFM (Recency, Frequency, Monetary) features
- Generate a proxy target for credit risk classification
- Train machine learning models
- Evaluate model performance
- Deploy a prediction API using FastAPI
- Implement CI/CD pipeline using GitHub Actions

---

## 📊 Dataset
The dataset contains transaction-level data with features such as:
- TransactionId
- CustomerId
- Amount
- TransactionStartTime
- ProductCategory
- ChannelId
- FraudResult (used indirectly for proxy labeling)

---

## 🧠 Feature Engineering

### Customer Aggregation Features
- Total transaction amount
- Average transaction amount
- Transaction count

### RFM Features
- **Recency**: Time since last transaction
- **Frequency**: Number of transactions
- **Monetary**: Total transaction value

---

## 🤖 Machine Learning Models

Two models were trained:

### 1. Logistic Regression
- Baseline model
- Simple and interpretable

### 2. Random Forest Classifier
- Better performance
- Handles non-linear relationships

---

## 📈 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## 🚀 API (FastAPI)

### Endpoint
 POST /predict


### Input Example
```json
{
  "total_amount": 1000,
  "avg_amount": 200,
  "transaction_count": 5
}
Output Example
{
  "risk_probability": 0.78,
  "is_high_risk": 1
}
🧪 CI/CD Pipeline

Implemented using GitHub Actions:

Linting with flake8
Testing with pytest
Automated environment setup
🛠 Tech Stack
Python 3.10
Pandas, NumPy
Scikit-learn
FastAPI
Joblib
GitHub Actions
📂 Project Structure
credit-risk-model/
│
├── src/
│   ├── data_processing.py
│   ├── train.py
│   ├── api/
│
├── tests/
├── .github/workflows/
├── model.pkl
├── requirements.txt
├── README.md
▶️ How to Run
Install dependencies
pip install -r requirements.txt
Train model
python -m src.train
Run API
uvicorn src.api.main:app --reload
✅ Status

✔ Data preprocessing
✔ Feature engineering
✔ RFM modeling
✔ Model training
✔ API deployment
✔ CI/CD pipeline

👨‍💻 Author

Credit Risk Modeling Project - Completed as part of ML Engineering Tasks


---

If you want next, I can help you:
- :contentReference[oaicite:0]{index=0}
- :contentReference[oaicite:1]{index=1}
- or :contentReference[oaicite:2]{index=2}