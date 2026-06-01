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

## 🏢 Business Understanding
Financial institutions face significant losses when customers default on loans or other financial obligations. Credit risk modeling helps organizations identify potentially risky customers before extending financial services.

The goal of this project is to build a machine learning system that analyzes customer transaction behavior and predicts the likelihood of a customer belonging to a high-risk segment.

The solution supports data-driven decision making in lending, customer monitoring, and risk management processes.

---

## ⚖️ Regulatory Considerations
Credit risk models operate in highly regulated environments and must ensure:

- Fair and unbiased decision making  
- Transparency and explainability of model predictions  
- Data privacy and customer confidentiality  
- Responsible use of machine learning systems  

This project uses transaction-level data and focuses on building a reproducible and interpretable risk prediction pipeline.

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

## 🤖 Modeling Approach
Since the dataset does not contain a direct default label, a proxy target was created using RFM-based clustering.

Steps:
1. Customers were grouped using **K-Means clustering**
2. Clusters were analyzed based on risk behavior
3. The highest-risk cluster was labeled as **high risk**
4. This label was used for supervised learning

Two models were trained:
- Logistic Regression (baseline model)
- Random Forest Classifier (advanced model)

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


### Input Example
```json
{
  "total_amount": 1000,
  "avg_amount": 200,
  "transaction_count": 5
}

{
  "risk_probability": 0.78,
  "is_high_risk": 1
}



🧪 CI/CD Pipeline

Implemented using GitHub Actions:

Code linting (flake8)
Unit testing (pytest)
Automated environment setup
Continuous integration workflow
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
│   │   └── main.py
│
├── tests/
├── .github/workflows/
├── model.pkl
├── requirements.txt
├── README.md
▶️ How to Run the Project
1. Install Dependencies
pip install -r requirements.txt
2. Train Model
python -m src.train
3. Run FastAPI Server
uvicorn src.api.main:app --reload
📊 Status

✔ Data preprocessing
✔ Feature engineering
✔ RFM modeling
✔ Model training
✔ API deployment
✔ CI/CD pipeline

👨‍💻 Author

Credit Risk Modeling Project — Machine Learning Engineering Implementation


---

If you want next upgrade, I can also help you:
- add **GitHub badges (build passing, Python version, etc.)**
- add **architecture diagram**
- or make it **portfolio-level (for internship/job)**