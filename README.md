# 💳 Credit Risk Probability Model — Bati Bank

[![CI Pipeline](https://github.com/yadakiya/credit-risk-model/actions/workflows/ci.yml/badge.svg)](https://github.com/yadakiya/credit-risk-model/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10-blue)

An end-to-end credit risk scoring service built for Bati Bank's buy-now-pay-later
partnership with an eCommerce platform. Transforms raw transaction data into a
customer risk probability, served via a REST API, with experiments tracked in
MLflow and quality gates enforced in CI.

## Business Problem

Bati Bank needs to decide, in real time, whether to extend a buy-now-pay-later
line of credit to a new customer — without a history of loan repayment to draw
on. The only signal available is the customer's past transaction behavior on
the partner eCommerce platform. This project turns that behavioral data into a
risk probability score the loan origination team can call via an API.

## Credit Scoring Business Understanding

**How does Basel II's emphasis on risk measurement shape the model?**
Basel II requires banks to demonstrate *how* a risk estimate was produced, not
just report a number — models used for capital and lending decisions must be
documented, auditable, and explainable to a regulator. That pushes this project
toward transparent feature engineering (aggregates and RFM values a
non-technical reviewer can inspect), fixed random seeds for reproducibility,
and logged experiments (MLflow) rather than an opaque, unversioned model.

**Why is a proxy target necessary, and what risk does that introduce?**
The raw transaction data has no "customer defaulted on a loan" label — there
were no loans yet. `is_high_risk` is therefore constructed from RFM behavior
(customers who transact rarely, recently stopped, and spend little are treated
as high risk). This is a *behavioral* proxy for a *credit* outcome, and the two
are not guaranteed to align: a disengaged shopper is not necessarily a bad
borrower. Business risk: the model could systematically deny credit to
customers who would have repaid reliably (false positives), or approve
customers who look active but have no repayment history at all (false
negatives). This assumption is a modeling choice, not ground truth, and should
be revisited once real repayment outcomes exist.

**Interpretable vs. high-performance models in a regulated context?**
Logistic Regression (optionally with Weight of Evidence-encoded features) is
easy to explain feature-by-feature to a risk committee or regulator, at some
cost to predictive accuracy on nonlinear patterns. Gradient boosting / random
forest models typically score higher on ranking metrics but require additional
tooling (e.g. SHAP) to explain individual decisions. This project trains both
and lets the metrics — not just raw accuracy — inform which is deployed;
Logistic Regression remains the safer default where regulatory sign-off on
explainability outweighs marginal AUC gains.

## Solution Overview

1. **Feature engineering** (`src/data_processing.py`) — aggregates raw
   transactions into customer-level features (total/average/count/std of
   transaction amount) and computes RFM values.
2. **Proxy target** — customers are K-Means clustered on scaled RFM values;
   the least-engaged cluster is labeled `is_high_risk`.
3. **Training** (`src/train.py`) — Logistic Regression and Random Forest are
   each trained as a single `sklearn.pipeline.Pipeline` (scaler + classifier),
   evaluated, and logged to MLflow; the best model by ROC-AUC is registered in
   the MLflow Model Registry and saved for serving.
4. **Serving** (`src/api/main.py`) — a FastAPI `/predict` endpoint loads that
   pipeline and returns a risk probability for a given customer's aggregate
   features.
5. **Explainability** (`src/explainability.py`) — SHAP-based explanations for
   both tree and linear models, answering *which features matter most
   globally* and *why the model made this specific prediction*.
6. **Dashboard** (`dashboard/app.py`) — a Streamlit app for the risk team:
   score a customer interactively and see the SHAP breakdown behind that
   score, or view portfolio-level feature importance and risk distribution.

## Engineering Notes

An earlier version of this pipeline trained the model on one-hot encoded
`CustomerId` values while the API sent aggregate transaction features at
prediction time — a feature mismatch that made every `/predict` call fail.
This has been fixed: training and serving now share a single typed config
(`src/config.py`) as the source of truth for which columns the model expects,
and `tests/test_data_processing.py::test_build_training_dataset_*` is a
regression test that would catch this class of bug if it recurred.

## Key Results

Metrics below are illustrative — regenerate with `python -m src.train` against
the real Xente dataset and update before final submission.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | — |
| Random Forest | — | — | — | — | — |

## Quick Start

```bash
git clone https://github.com/yadakiya/credit-risk-model
cd credit-risk-model
pip install -r requirements.txt

# Place the Xente dataset at data/raw/data.csv, then:
python -m src.train          # trains, tracks in MLflow, saves model.pkl
uvicorn src.api.main:app --reload
```

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"total_amount": 15000, "avg_amount": 750, "transaction_count": 20, "std_amount": 320.5}'
```

```json
{"risk_probability": 0.12, "is_high_risk": 0}
```

## Dashboard

An interactive Streamlit dashboard for the risk team: score a customer and
see the SHAP breakdown behind that score, or view portfolio-level feature
importance and risk distribution.

```bash
streamlit run dashboard/app.py
```

Requires `model.pkl` (from `python -m src.train`). The portfolio view also
picks up `data/processed/training_dataset.csv` if present (also produced by
`python -m src.train`) for the risk-distribution chart; without it, the
dashboard still works for single-customer scoring and global feature
importance.

## Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml     # lint + test on every push/PR to main
├── notebooks/eda.ipynb          # exploratory analysis
├── dashboard/
│   └── app.py                   # Streamlit risk dashboard
├── src/
│   ├── config.py                # typed config: single source of truth
│   ├── data_processing.py       # feature engineering + proxy target
│   ├── train.py                 # training, MLflow tracking, registry
│   ├── explainability.py        # SHAP global + per-prediction explanations
│   └── api/
│       ├── main.py              # FastAPI app
│       └── pydantic_models.py   # request/response schemas
├── tests/
│   ├── test_data_processing.py
│   ├── test_api.py
│   └── test_explainability.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Running Tests

```bash
pytest -v
flake8 src tests
```

## Docker

```bash
docker compose up --build
```

## Demo

_Add screenshots here before final submission: `streamlit run dashboard/app.py`,
screenshot the "Score a customer" tab (with the SHAP chart visible) and the
"Portfolio overview" tab, plus the MLflow UI (`mlflow ui`) and the passing
GitHub Actions run._

## Technical Details

- **Data**: Xente eCommerce transaction records (Kaggle Xente Challenge).
- **Proxy target**: K-Means (k=3, `random_state=42`) on scaled RFM features;
  least-engaged cluster → `is_high_risk=1`.
- **Models**: Logistic Regression, Random Forest — both wrapped as a single
  `sklearn.pipeline.Pipeline` so scaling and prediction never drift apart.
- **Tracking**: MLflow experiment `credit-risk-model`, model registry name
  `credit-risk-classifier`.
- **Explainability**: SHAP (`TreeExplainer` for Random Forest, kernel-based
  `Explainer` for Logistic Regression), surfaced in both the dashboard and
  available for the API to call.

## Future Improvements

- Weight of Evidence / Information Value feature encoding
- Loan amount/duration recommendation model (beyond binary risk classification)
- Deploy the dashboard (Streamlit Community Cloud / internal server) for the risk team, rather than local-only
- Hook the API up to a live MLflow tracking server so `/predict` always serves the latest registered model instead of a static `model.pkl`

## Author

Yadeni Getu
  — Credit Risk Modeling Project, Analytics Engineering.
