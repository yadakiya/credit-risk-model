# Credit Scoring Business Understanding

## 1. Basel II and the Need for Interpretability

The Basel II Accord emphasizes accurate risk measurement, transparency, and proper documentation in credit risk modeling. Financial institutions are expected to justify how their models make predictions and demonstrate that risk estimates are reliable and reproducible.

Because credit decisions directly affect customers and regulatory compliance, interpretable models are highly valuable. Models such as Logistic Regression combined with Weight of Evidence (WoE) transformations allow banks to clearly explain how input variables influence the final credit risk score.

In this project, interpretability is important because:

* The bank must explain lending decisions
* Regulators require model transparency
* Risk teams need to validate model behavior
* The model must support monitoring and auditing over time

Therefore, all preprocessing, feature engineering, and modeling steps must be reproducible and well documented.

---

## 2. Why a Proxy Target Variable is Necessary

The provided dataset does not contain a direct “default” label indicating whether a customer failed to repay a loan. Since supervised machine learning models require target labels, a proxy variable must be engineered from available behavioral transaction data.

This project uses RFM (Recency, Frequency, Monetary) analysis to identify customer engagement patterns. Customers who transact infrequently, spend less money, and remain inactive for long periods are treated as higher-risk customers.

The proxy target variable introduces several business risks:

* The proxy may not perfectly represent real default behavior
* Some low-activity customers may still repay loans successfully
* The clustering process may create noisy labels
* Model predictions depend heavily on assumptions made during proxy construction

Because of these risks, the proxy target should be treated as an approximation rather than ground truth.

---

## 3. Trade-offs Between Interpretable and High-Performance Models

There is an important trade-off between model interpretability and predictive performance in regulated financial systems.

### Logistic Regression with WoE

Advantages:

* Highly interpretable
* Easy to explain to regulators
* Stable and simple
* Works well with scorecards

Disadvantages:

* May miss complex nonlinear relationships
* Lower predictive performance compared to ensemble models

### Gradient Boosting Models (XGBoost/LightGBM)

Advantages:

* High predictive accuracy
* Captures nonlinear interactions
* Strong performance on tabular datasets

Disadvantages:

* Harder to interpret
* More difficult to explain to regulators
* Requires additional explainability techniques

In regulated environments such as banking, institutions often balance both objectives by comparing interpretable baseline models against more advanced machine learning models.
