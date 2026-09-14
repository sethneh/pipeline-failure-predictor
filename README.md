# 🔮 Pipeline Failure Predictor

### Predict data pipeline failures before they happen using Machine Learning

**Can we predict whether a data pipeline run will fail before it actually fails?**

This project uses machine learning to predict pipeline failures from operational metadata such as **run duration, row counts, null rates, schema changes, and upstream pipeline status**.

The project uses synthetic Airflow/dbt-style metadata to simulate real-world data engineering scenarios.

---

## 🎯 Problem

Data pipeline failures are expensive.

A failed pipeline can cause:

* Delayed dashboards and reports
* Missed SLAs
* Downstream pipeline failures
* On-call alerts and manual investigation
* Data quality issues reaching consumers

Traditional monitoring usually tells you **after something has failed**.

This project explores a different question:

> **Can we identify a pipeline that is likely to fail before the failure happens?**

---

## 🏗️ Solution

The project generates realistic pipeline-run metadata and trains classification models to predict:

**`OK`** → pipeline is expected to succeed
**`FAIL`** → pipeline is likely to fail

### High-level architecture

```text
Synthetic Pipeline Metadata
          │
          ▼
   Data Generation
          │
          ▼
   Feature Engineering
          │
          ▼
 Train / Test Split
          │
          ▼
 ┌─────────────────────┐
 │ ML Classification   │
 │                     │
 │ Logistic Regression │
 │ Random Forest       │
 └─────────────────────┘
          │
          ▼
 Model Evaluation
          │
          ├── Precision
          ├── Recall
          ├── F1 Score
          └── ROC-AUC
          │
          ▼
 Feature Importance
          │
          ▼
 "Will this run fail?"
```

---

## 📊 Features

The model uses operational signals that a real data platform could capture:

| Feature         |       Example | Why it matters                              |
| --------------- | ------------: | ------------------------------------------- |
| Run duration    |        42 min | Unusual duration can indicate problems      |
| Row count       |          1.2M | Sudden changes may indicate upstream issues |
| Null rate       |           18% | Data quality degradation                    |
| Schema change   |           Yes | Can break downstream processing             |
| Upstream status |        Failed | Strong indicator of downstream failure      |
| Retry count     |             2 | May indicate instability                    |
| Pipeline type   | dbt / Airflow | Operational context                         |

The dataset contains **5,000 synthetic pipeline runs**, with failures intentionally introduced to simulate an imbalanced production environment.

---

## 🤖 Machine Learning

This is a **binary classification** problem.

The model predicts:

```text
0 → OK
1 → FAIL
```

The project compares classification approaches such as:

* Logistic Regression
* Random Forest

### Why not just use accuracy?

Failures are relatively uncommon.

For example:

```text
93% OK
 7% FAIL
```

A model that predicts **OK every time** could achieve 93% accuracy while being completely useless.

Therefore, the project focuses on:

### Precision

> Of the runs we predicted would fail, how many actually failed?

Important for reducing **false alarms**.

### Recall

> Of the runs that actually failed, how many did we detect?

Important for preventing **missed failures**.

### F1 Score

Balances precision and recall.

### ROC-AUC

Measures how well the model separates successful and failed pipeline runs across different probability thresholds.

---

## ⚠️ The Alert-Fatigue Problem

In production, predicting failures isn't enough.

Imagine:

```text
100 pipeline runs
       │
       ▼
20 alerts
       │
       ├── 5 real failures
       └── 15 false alarms
```

The engineering team may eventually start ignoring the alerts.

That's why this project explores the tradeoff between:

**High Recall**

→ Catch more failures
→ More false alarms

**High Precision**

→ Fewer false alarms
→ Potentially miss some failures

The ideal threshold depends on the team's **alert budget and tolerance for missed failures**.

---

## 🛡️ Preventing Data Leakage

One important ML engineering principle demonstrated in this project is **avoiding data leakage**.

Preprocessing such as scaling is performed inside a scikit-learn `Pipeline`.

Conceptually:

```text
Training Data
     │
     ▼
Scaler ──► Model
     │
     ▼
Predictions

Test Data
     │
     ▼
Same fitted scaler
     │
     ▼
Model
```

The test data is never used to calculate training transformations.

This ensures evaluation more closely represents how the model will behave on unseen production data.

---

## 🔍 Interpretability

A prediction is much more useful to an engineer when the system can explain **why** it thinks a failure is likely.

For example:

```text
Prediction: FAIL
Probability: 87%

Important signals:
────────────────────────────
Upstream status       ██████████
Schema change         ████████
Run duration          ██████
Null rate             ████
Row count             ██
```

The goal is to connect ML predictions back to operational signals that data engineers understand.

For example:

> "This pipeline has a high failure probability because an upstream dependency failed, the schema changed, and the current runtime is significantly higher than normal."

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd pipeline-failure-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate synthetic data

```bash
python generate_data.py
```

This creates:

```text
pipeline_runs.csv
```

containing 5,000 synthetic pipeline runs.

### 4. Train the models

```bash
python train.py
```

The script trains the models and prints evaluation metrics such as:

```text
--- RandomForestClassifier ---

accuracy : 0.XXX
precision: 0.XXX
recall   : 0.XXX
F1       : 0.XXX
ROC-AUC  : 0.XXX
```

---

## 📁 Project Structure

```text
pipeline-failure-predictor/
│
├── generate_data.py       # Generates synthetic pipeline metadata
├── train.py               # Trains and evaluates ML models
├── pipeline_runs.csv      # Generated dataset
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .gitignore
```

---

## 💡 What This Project Demonstrates

### Data Engineering

* Pipeline operational metadata
* Data quality signals
* Upstream/downstream dependencies
* Airflow/dbt concepts
* Failure monitoring

### Machine Learning

* Binary classification
* Imbalanced datasets
* Train/test splitting
* Feature preprocessing
* Random Forest
* Precision / Recall / F1
* ROC-AUC
* Feature importance

### ML Engineering

* Preventing data leakage
* Reproducible training
* Model evaluation
* Interpretable predictions
* Production-oriented thinking

---

## 🔮 Future Improvements

This project can evolve from a portfolio ML experiment into a small **data observability platform**.

### 1. SLA Prediction

Instead of predicting only:

```text
FAIL / OK
```

predict:

```text
Expected runtime: 37 minutes
```

This becomes a regression problem.

### 2. Intelligent Alert Thresholds

Instead of using a fixed probability such as:

```text
FAIL probability > 50%
```

select the threshold based on the team's acceptable alert volume.

### 3. Real Airflow / dbt Metadata

Replace synthetic data with real pipeline metadata from:

```text
Airflow metadata DB
        +
dbt artifacts
        +
Data quality metrics
```

### 4. Data Drift Monitoring

Monitor changes in pipeline behavior and automatically trigger retraining when the underlying data distribution changes.

### 5. Real-Time Prediction API

Deploy the model behind a FastAPI service:

```text
Airflow / dbt
     │
     ▼
Pipeline Run
     │
     ▼
FastAPI
     │
     ▼
ML Model
     │
     ▼
Failure Probability
     │
     ▼
Alert / Dashboard
```

Example:

```json
{
  "pipeline": "daily_customer_load",
  "failure_probability": 0.87,
  "prediction": "FAIL"
}
```

---

## ⭐ Why I Built This

Most beginner ML projects focus on problems such as house prices, customer churn, or movie recommendations.

I wanted to build something closer to the problems I have worked with as a **data engineering and data platform professional**.

This project combines:

**Data Engineering + Machine Learning + Data Observability**

to explore how ML can move data platforms from:

> **"Tell me when the pipeline failed."**

to:

> **"Tell me which pipeline is likely to fail — and why."**

---

## 🛠️ Tech Stack

```text
Python
scikit-learn
Pandas
NumPy
Airflow / dbt concepts
Machine Learning
Git / GitHub
```

---

## 📌 Project Status

**Current:** ML classification prototype using synthetic pipeline metadata.

**Next:** Add probability-based alerting, model explainability, and a FastAPI prediction service.
