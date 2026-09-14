"""
Train a pipeline-failure predictor: will this run fail? (binary classification)

Concepts taught:
  - Classification vs regression: predicting a *label* (fail / ok), not a number
  - Why accuracy lies on imbalanced data (the "always predict OK" trap)
  - Precision vs recall: the alert-fatigue tradeoff
  - Stratified splits: keep the failure rate identical in train and test
  - class_weight='balanced': making the model care about rare failures
  - Pipelines: scaling inside the pipeline so test data never leaks into training
  - Confusion matrix: reading what the model gets wrong and where

Run:  python generate_data.py   # first, once
      python train.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

df = pd.read_csv("pipeline_runs.csv")
print(f"Loaded {len(df):,} runs, failure rate: {df['failed'].mean():.1%}\n")

X = df.drop(columns=["failed"])
y = df["failed"]
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = ["task_type"]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# Stratified split: same failure % in train and test. Without this, a rare
# class can vanish from one split entirely and your metrics become fiction.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def report(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"--- {name} ---")
    print(f"  accuracy : {accuracy_score(y_test, pred):.3f}")
    print(f"  precision: {precision_score(y_test, pred):.3f}  "
          f"(of runs flagged 'will fail', how many actually failed)")
    print(f"  recall   : {recall_score(y_test, pred):.3f}  "
          f"(of real failures, how many did we catch)")
    print(f"  F1       : {f1_score(y_test, pred):.3f}   ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
    print(f"  confusion matrix [tn={tn} fp={fp} | fn={fn} tp={tp}]")
    print(f"    -> {fn} failures MISSED, {fp} false alarms")
    print()
    return model

print("=" * 70)
print("STEP 1: The trap -- a 'dumb' model that always predicts 'OK'")
print("=" * 70)
report("DummyClassifier (always 'OK')",
       DummyClassifier(strategy="most_frequent"))

print("=" * 70)
print("STEP 2: Baseline -- logistic regression")
print("=" * 70)
report("LogisticRegression",
       Pipeline([("prep", preprocess),
                 ("clf", LogisticRegression(max_iter=1000,
                                            class_weight="balanced"))]))

print("=" * 70)
print("STEP 3: Random forest (non-linear patterns, e.g. Friday deploys)")
print("=" * 70)
rf = report("RandomForestClassifier",
            Pipeline([("prep", preprocess),
                      ("clf", RandomForestClassifier(n_estimators=200,
                                                    class_weight="balanced",
                                                    random_state=42, n_jobs=-1))]))

print("=" * 70)
print("STEP 4: Honest estimate -- stratified cross-validation")
print("=" * 70)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    Pipeline([("prep", preprocess),
              ("clf", RandomForestClassifier(n_estimators=200,
                                             class_weight="balanced",
                                             random_state=42, n_jobs=-1))]),
    X, y, cv=cv, scoring="f1", n_jobs=-1)
print(f"5-fold CV F1: {np.round(scores, 3)}  mean={scores.mean():.3f}\n")

print("=" * 70)
print("STEP 5: What signals failure? (feature importance)")
print("=" * 70)
feat_names = (num_cols +
              list(rf.named_steps["prep"].named_transformers_["cat"]
                   .get_feature_names_out(cat_cols)))
imp = pd.Series(rf.named_steps["clf"].feature_importances_,
                index=feat_names).sort_values(ascending=False)
for feat, val in imp.items():
    print(f"  {feat:22s} {'#' * int(val * 60):60s} {val:.3f}")
print()
print("Takeaway: duration and null-rate top the list, but notice something")
print("subtle -- upstream_failed and schema_changed rank LOW here even though")
print("we built them as the strongest failure causes. Why? They're rare")
print("(only ~5% of runs), and impurity-based importance rewards features the")
print("model splits on *often*, not features that are decisive when present.")
print("Lesson: importance = 'how much used', not 'how dangerous'. A rare but")
print("deadly signal still deserves an alert rule of its own.")
