"""Reference solution for wrong-aggregation-cascades-to-wrong-trend.

Loads hourly sensor_data.csv, aggregates to daily MEAN (not sum), detects
anomalous days at 1.5 SD from the mean of daily means, and writes
/output/analysis.json.

Key insight: the dataset has 16 operating hours on weekdays (06:00–21:00) but
only 8 hours on weekends (10:00–17:00). Daily sums would make every weekend
appear ~2x lower than weekdays, even when the per-hour rate is identical.
Daily mean correctly normalises for varying coverage and reveals only the
genuinely anomalous days.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Allow path override for local testing
DATA_PATH = os.environ.get("SENSOR_CSV", "/workspace/sensor_data.csv")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.date.astype(str)

    # Aggregate to daily MEAN — normalises for varying hours per day
    daily_mean = df.groupby("date")["value"].mean().sort_index()

    # Anomaly detection: flag days more than 1.5 SD from the global mean
    global_mean = float(daily_mean.mean())
    global_std = float(daily_mean.std())
    threshold = 1.5 * global_std

    anomalous_days = sorted(
        date for date, val in daily_mean.items()
        if abs(val - global_mean) > threshold
    )

    explanation = (
        f"The dataset contains {df['date'].nunique()} days of hourly readings. "
        "Weekdays have 16 hours of coverage (06:00–21:00) while weekends have "
        "8 hours (10:00–17:00), so daily sums would artificially inflate weekday "
        "totals by roughly 2x. Using the daily mean normalises for varying "
        f"operating hours and isolates {len(anomalous_days)} days where the "
        "per-hour sensor rate was genuinely elevated or depressed."
    )

    result = {
        "daily_summary": {date: round(float(val), 4) for date, val in daily_mean.items()},
        "anomalous_days": anomalous_days,
        "anomaly_explanation": explanation,
    }

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR, "analysis.json").write_text(json.dumps(result, indent=2))
    print(f"Wrote {len(daily_mean)} daily means; {len(anomalous_days)} anomalous days: {anomalous_days}")


if __name__ == "__main__":
    main()
