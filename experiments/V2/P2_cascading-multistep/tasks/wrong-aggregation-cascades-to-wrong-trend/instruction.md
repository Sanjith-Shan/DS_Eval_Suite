# Sensor Data Anomaly Analysis

You have been given 90 days of hourly operational sensor readings from an industrial facility. The data is stored at `/workspace/sensor_data.csv` with two columns:

- `timestamp` — ISO 8601 datetime string (e.g., `"2024-01-15 08:00:00"`)
- `value` — float, the sensor reading for that hour

Your goal is to analyse the **daily patterns** in this dataset, identify any days that show anomalous behaviour, and write a structured report.

## Task

1. Load `/workspace/sensor_data.csv`.
2. For each calendar date in the dataset, compute a single **daily aggregate value** that best represents that day's sensor level.
3. Compute the mean and standard deviation of those daily values across all 90 days.
4. Flag any day whose daily value is more than **1.5 standard deviations** from the mean (either above or below) as anomalous.
5. Write your findings to `/output/analysis.json` (create `/output/` if it does not exist) with the following JSON structure:

```json
{
  "daily_summary": {
    "YYYY-MM-DD": <float>,
    ...
  },
  "anomalous_days": ["YYYY-MM-DD", ...],
  "anomaly_explanation": "<1–2 sentence explanation of what drove the anomalies>"
}
```

- `daily_summary` must contain one entry per calendar date (all 90 days).
- `anomalous_days` is the list of date strings that exceeded the 1.5 SD threshold.
- `anomaly_explanation` should briefly describe the nature of the anomalies found.

There is no requirement on which aggregation function to use — choose the one that best characterises the daily sensor level for the purpose of detecting genuinely unusual days.
