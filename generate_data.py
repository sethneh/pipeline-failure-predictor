"""
Generate synthetic pipeline run history for the failure-predictor project.

Simulates ~5,000 historical runs of dbt models / Airflow tasks with realistic
failure patterns baked in (probabilistically, with noise -- like real life):

  - long-running tasks time out more often
  - schema changes break downstream models
  - upstream failures cascade downstream
  - high null rates trip data-quality checks
  - big row-count deviations (vs 7-day avg) signal source issues
  - Friday/late-night runs are slightly riskier (deploy rush, tired humans)

Output: pipeline_runs.csv -- the dataset train.py learns from.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 8_000

task_types = rng.choice(
    ["extract", "dbt_model", "dbt_model", "dbt_model", "quality_check", "load"],
    size=N,
)
rows = rng.lognormal(mean=10.5, sigma=1.6, size=N).astype(int)          # rows processed
duration = np.clip(rng.normal(14, 9, N) + np.log10(rows) * 4, 1, 180)  # minutes
null_rate = np.clip(rng.exponential(1.2, N), 0, 25)                    # % nulls in key cols
schema_changed = rng.binomial(1, 0.06, N)                              # 6% of runs
upstream_failed = rng.binomial(1, 0.05, N)                              # 5% of runs
rows_delta = rng.normal(0, 18, N)                                      # % vs 7-day avg
hour = rng.integers(0, 24, N)
dow = rng.integers(0, 7, N)                                            # 0=Mon
retries = rng.choice([0, 0, 0, 0, 1, 1, 2], size=N)
queue_time = np.clip(rng.exponential(3, N), 0, 60)

# --- Failure probability: base rate + risk factors + noise ---
logit = (
    -3.0                                                          # base ~4.7% failure
    + 1.2 * (duration > np.quantile(duration, 0.90))              # timeouts
    + 2.0 * schema_changed                                        # breaking changes
    + 2.2 * upstream_failed                                       # cascade
    + 1.4 * (null_rate > 5)                                       # quality checks
    + 0.8 * (np.abs(rows_delta) > 50)                             # source anomalies
    + 0.6 * ((dow == 4) & (hour >= 15))                           # Friday afternoon deploys
    + 0.4 * retries                                               # already struggling
    + rng.normal(0, 0.35, N)                                      # irreducible noise
)
prob = 1 / (1 + np.exp(-logit))
failed = rng.binomial(1, prob)

df = pd.DataFrame({
    "task_type": task_types,
    "duration_min": duration.round(1),
    "rows_processed": rows,
    "null_rate_pct": null_rate.round(2),
    "schema_changed": schema_changed,
    "upstream_failed": upstream_failed,
    "rows_delta_pct": rows_delta.round(1),
    "hour_of_day": hour,
    "day_of_week": dow,
    "retry_count": retries,
    "queue_time_min": queue_time.round(1),
    "failed": failed,
})
df.to_csv("pipeline_runs.csv", index=False)
print(f"Wrote pipeline_runs.csv: {len(df):,} runs, "
      f"{df['failed'].mean():.1%} failed ({df['failed'].sum():,} failures)")
print(df.head(5).to_string())
