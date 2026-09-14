# Pipeline Failure Predictor

> Built with AI assistance as a hands-on machine learning project.

**Will this pipeline run fail?** — ML for data observability. Trained on
synthetic dbt/Airflow run metadata (durations, row counts, null rates, schema
changes, upstream status), it predicts run failures *before* they happen and
explains which signals matter most.

## Why this project
Most ML portfolios are house prices and churn. This one is different: it
applies machine learning to the operational reality of data engineering —
something only a data engineer would think to build, and only ML can solve
well at scale.

## Quickstart

```bash
pip install -r requirements.txt
python generate_data.py   # creates pipeline_runs.csv (5,000 synthetic runs)
python train.py           # trains & evaluates the models
```

## What it teaches

- **Classification**: predicting a label (fail / ok) instead of a number
- **The accuracy trap**: a model that always predicts "OK" scores 93% accuracy
  and is completely useless — why precision/recall/F1 matter
- **Precision vs recall**: the alert-fatigue tradeoff (false alarms vs missed
  failures) every on-call engineer lives with
- **Imbalanced data**: failures are uncommon (~12%); `class_weight='balanced'` and
  stratified splits keep the model honest
- **No leakage**: scaling inside a `Pipeline` so test data never contaminates
  training
- **Interpretability**: feature importances that match real operational
  intuition (upstream failures, schema changes, extreme durations)

## Sample output

```
--- RandomForestClassifier ---
  accuracy : 0.XXX
  precision: 0.XXX   (of runs flagged 'will fail', how many actually failed)
  recall   : 0.XXX   (of real failures, how many did we catch)
  F1       : 0.XXX   ROC-AUC: 0.XXX
```

## Ideas to extend

1. **SLA regression**: predict *how long* a run will take, not just fail/ok
2. **Threshold tuning**: pick the probability cutoff from the precision-recall
   curve to match your team's alert budget
3. **Real data**: point it at your Airflow/dbt metadata DB instead of
   synthetic data
4. **Drift monitoring**: retrain weekly; alert when feature importances shift
5. **Serve it**: wrap the model in a FastAPI endpoint that scores runs live
