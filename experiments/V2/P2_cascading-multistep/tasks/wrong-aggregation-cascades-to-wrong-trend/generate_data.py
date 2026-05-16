"""Generate sensor_data.csv for wrong-aggregation-cascades-to-wrong-trend task.

Design:
- 90 days of hourly sensor readings starting 2024-01-01
- Weekdays (Mon-Fri): hours 6-21 inclusive (16 rows/day)
- Weekends (Sat-Sun): hours 10-17 inclusive (8 rows/day)
- Hourly rate ~ Normal(mean=50, sd=2) — IDENTICAL across all day types
- 3 planted genuine anomalies:
    Day index 15 (2024-01-16, Tuesday): hourly mean shifted to 75
    Day index 47 (2024-02-17, Saturday): hourly mean shifted to 25
    Day index 72 (2024-03-13, Wednesday): hourly mean shifted to 75
- Hourly noise sd reduced to 2.0 so the bimodal sum distribution is well-separated
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

START_DATE = pd.Timestamp("2024-01-01")
N_DAYS = 90

# Planted anomaly day indices (0-based from start date)
ANOMALY_INDICES = {15: 75.0, 47: 25.0, 72: 75.0}

rows = []
for day_offset in range(N_DAYS):
    day = START_DATE + pd.Timedelta(days=day_offset)
    dow = day.dayofweek  # 0=Mon, 6=Sun
    is_weekend = dow >= 5

    if is_weekend:
        hours = list(range(10, 18))  # 10, 11, ..., 17 => 8 hours
    else:
        hours = list(range(6, 22))   # 6, 7, ..., 21 => 16 hours

    # Check if this day is a planted anomaly
    if day_offset in ANOMALY_INDICES:
        mean_val = ANOMALY_INDICES[day_offset]
    else:
        mean_val = 50.0

    for h in hours:
        ts = day + pd.Timedelta(hours=h)
        value = np.random.normal(loc=mean_val, scale=2.0)
        rows.append({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "value": round(value, 4)})

df = pd.DataFrame(rows)

# Save
out_path = Path(__file__).parent / "environment" / "sensor_data.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")

# ── Diagnostic stats ──────────────────────────────────────────────────────────
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date
df["dow"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = df["dow"] >= 5

daily_sum  = df.groupby("date")["value"].sum()
daily_mean = df.groupby("date")["value"].mean()
date_is_weekend = df.groupby("date")["is_weekend"].first()

n_weekday_days = (~date_is_weekend).sum()
n_weekend_days = date_is_weekend.sum()

print(f"\nDay counts over {N_DAYS} days:")
print(f"  Weekdays : {n_weekday_days}")
print(f"  Weekends : {n_weekend_days}")

print(f"\nMedian DAILY SUM by day type:")
print(f"  Weekday median sum : {daily_sum[~date_is_weekend].median():.2f}")
print(f"  Weekend median sum : {daily_sum[ date_is_weekend].median():.2f}")

print(f"\nMedian DAILY MEAN by day type:")
print(f"  Weekday median mean : {daily_mean[~date_is_weekend].median():.2f}")
print(f"  Weekend median mean : {daily_mean[ date_is_weekend].median():.2f}")

# Anomaly detection — SUM-based
sum_mean = daily_sum.mean()
sum_std  = daily_sum.std()
flagged_sum = daily_sum[abs(daily_sum - sum_mean) > 1.5 * sum_std]
print(f"\nSUM-based anomaly detection (1.5σ):")
print(f"  Global sum mean={sum_mean:.2f}, std={sum_std:.2f}")
print(f"  Days flagged : {len(flagged_sum)}")
print(f"  Flagged dates (first 10): {sorted(flagged_sum.index)[:10]}")

# Anomaly detection — MEAN-based
mean_mean = daily_mean.mean()
mean_std  = daily_mean.std()
flagged_mean = daily_mean[abs(daily_mean - mean_mean) > 1.5 * mean_std]
print(f"\nMEAN-based anomaly detection (1.5σ):")
print(f"  Global daily-mean mean={mean_mean:.2f}, std={mean_std:.2f}")
print(f"  Days flagged : {len(flagged_mean)}")
print(f"  Flagged dates: {sorted(flagged_mean.index)}")

# Confirm planted anomaly dates
planted_dates = []
for idx, shifted_mean in ANOMALY_INDICES.items():
    d = (START_DATE + pd.Timedelta(days=idx)).date()
    dow_name = (START_DATE + pd.Timedelta(days=idx)).strftime("%A")
    planted_dates.append(str(d))
    print(f"\nPlanted anomaly: day_index={idx}  date={d}  ({dow_name})  target_mean={shifted_mean}")

print(f"\nPlanted anomaly dates (ground truth): {planted_dates}")
