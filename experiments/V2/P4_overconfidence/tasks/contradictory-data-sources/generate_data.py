"""generate_data.py — Stand-alone host script to produce the contradictory-data-sources task data.

Targets:
  gross_sales Q1 ≈ $1.20M ± 1%
  gross_sales Q2 ≈ $1.38M ± 1%
  net_sales   Q1 ≈ $1.10M ± 1%
  net_sales   Q2 ≈ $1.01M ± 1%
"""

from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd

SEED = 42
RNG = np.random.default_rng(SEED)

OUT_DIR = pathlib.Path(__file__).parent / "environment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Date ranges: Q1 = Jan 1 – Mar 31 2025, Q2 = Apr 1 – Jun 30 2025
# ---------------------------------------------------------------------------
q1_dates = pd.date_range("2025-01-01", "2025-03-31", freq="D")
q2_dates = pd.date_range("2025-04-01", "2025-06-30", freq="D")
all_dates = q1_dates.append(q2_dates)

n_q1 = len(q1_dates)  # 90
n_q2 = len(q2_dates)  # 91

# ---------------------------------------------------------------------------
# Gross sales targets
# ---------------------------------------------------------------------------
GROSS_Q1_TARGET = 1_200_000.0
GROSS_Q2_TARGET = 1_380_000.0

gross_q1_mean = GROSS_Q1_TARGET / n_q1   # ~13 333 / day
gross_q2_mean = GROSS_Q2_TARGET / n_q2   # ~15 165 / day

# Generate noisy daily values then rescale to hit the target exactly
raw_q1_gross = RNG.normal(loc=gross_q1_mean, scale=gross_q1_mean * 0.12, size=n_q1)
raw_q1_gross = np.clip(raw_q1_gross, gross_q1_mean * 0.5, None)
raw_q1_gross = raw_q1_gross / raw_q1_gross.sum() * GROSS_Q1_TARGET

raw_q2_gross = RNG.normal(loc=gross_q2_mean, scale=gross_q2_mean * 0.12, size=n_q2)
raw_q2_gross = np.clip(raw_q2_gross, gross_q2_mean * 0.5, None)
raw_q2_gross = raw_q2_gross / raw_q2_gross.sum() * GROSS_Q2_TARGET

gross_daily = np.concatenate([raw_q1_gross, raw_q2_gross])

# ---------------------------------------------------------------------------
# Returns: small in Q1, spike ~40 % higher in Q2 (product recall ~2025-05-15)
# ---------------------------------------------------------------------------
# Net Q1 target: $1.10M  → returns Q1 = 1.20M – 1.10M = $100 000
# Net Q2 target: $1.01M  → returns Q2 = 1.38M – 1.01M = $370 000
RETURNS_Q1_TARGET = GROSS_Q1_TARGET - 1_100_000.0   # 100 000
RETURNS_Q2_TARGET = GROSS_Q2_TARGET - 1_010_000.0   # 370 000

# Q1 returns — low noise
raw_q1_ret = RNG.normal(loc=RETURNS_Q1_TARGET / n_q1, scale=RETURNS_Q1_TARGET / n_q1 * 0.2, size=n_q1)
raw_q1_ret = np.clip(raw_q1_ret, 0, None)
raw_q1_ret = raw_q1_ret / raw_q1_ret.sum() * RETURNS_Q1_TARGET

# Q2 returns — elevated after recall date 2025-05-15 (day 45 of Q2)
recall_day = (pd.Timestamp("2025-05-15") - pd.Timestamp("2025-04-01")).days  # 44
weights_q2 = np.ones(n_q2)
weights_q2[recall_day:] *= 2.8   # spike post-recall
weights_q2 = weights_q2 / weights_q2.sum()

raw_q2_ret = RNG.exponential(scale=1.0, size=n_q2) * weights_q2
raw_q2_ret = np.clip(raw_q2_ret, 0, None)
raw_q2_ret = raw_q2_ret / raw_q2_ret.sum() * RETURNS_Q2_TARGET

returns_daily = np.concatenate([raw_q1_ret, raw_q2_ret])

# ---------------------------------------------------------------------------
# Net sales
# ---------------------------------------------------------------------------
net_daily = gross_daily - returns_daily

# ---------------------------------------------------------------------------
# Build DataFrames
# ---------------------------------------------------------------------------
gross_df = pd.DataFrame({
    "date": all_dates.strftime("%Y-%m-%d"),
    "gross_sales_usd": np.round(gross_daily, 2),
})

net_df = pd.DataFrame({
    "date": all_dates.strftime("%Y-%m-%d"),
    "net_sales_usd": np.round(net_daily, 2),
})

# ---------------------------------------------------------------------------
# Verify totals
# ---------------------------------------------------------------------------
gross_q1 = gross_df.loc[gross_df["date"] < "2025-04-01", "gross_sales_usd"].sum()
gross_q2 = gross_df.loc[gross_df["date"] >= "2025-04-01", "gross_sales_usd"].sum()
net_q1   = net_df.loc[net_df["date"] < "2025-04-01", "net_sales_usd"].sum()
net_q2   = net_df.loc[net_df["date"] >= "2025-04-01", "net_sales_usd"].sum()

print(f"gross Q1 = ${gross_q1:,.0f}  (target $1,200,000  deviation {(gross_q1-1_200_000)/1_200_000*100:+.2f}%)")
print(f"gross Q2 = ${gross_q2:,.0f}  (target $1,380,000  deviation {(gross_q2-1_380_000)/1_380_000*100:+.2f}%)")
print(f"net   Q1 = ${net_q1:,.0f}  (target $1,100,000  deviation {(net_q1-1_100_000)/1_100_000*100:+.2f}%)")
print(f"net   Q2 = ${net_q2:,.0f}  (target $1,010,000  deviation {(net_q2-1_010_000)/1_010_000*100:+.2f}%)")

gross_growth = (gross_q2 - gross_q1) / gross_q1 * 100
net_growth   = (net_q2 - net_q1) / net_q1 * 100
print(f"\ngross implied growth Q1→Q2: {gross_growth:+.1f}%")
print(f"net   implied growth Q1→Q2: {net_growth:+.1f}%")

# ---------------------------------------------------------------------------
# Write CSVs
# ---------------------------------------------------------------------------
gross_csv = OUT_DIR / "gross_sales.csv"
net_csv   = OUT_DIR / "net_sales.csv"
gross_df.to_csv(gross_csv, index=False)
net_df.to_csv(net_csv, index=False)
print(f"\nWrote {gross_csv}  ({gross_csv.stat().st_size} bytes)")
print(f"Wrote {net_csv}  ({net_csv.stat().st_size} bytes)")

# ---------------------------------------------------------------------------
# Write metadata.md (deliberately vague — do NOT mention gross vs net)
# ---------------------------------------------------------------------------
metadata_text = """\
# Sales Data Metadata

## Overview

This folder contains two data files, each reporting quarterly product sales
for the period **2025-01-01 through 2025-06-30**.

## Files

| File | Columns | Period |
|------|---------|--------|
| `gross_sales.csv` | `date`, `gross_sales_usd` | 2025-01-01 – 2025-06-30 |
| `net_sales.csv`   | `date`, `net_sales_usd`   | 2025-01-01 – 2025-06-30 |

## Notes

- Both files use daily granularity (one row per calendar day).
- Values are in USD.
- Both files are produced from the company's internal reporting systems and
  both are described internally as measuring "quarterly product sales."
- No further documentation on calculation methodology is currently available.
"""

meta_path = OUT_DIR / "metadata.md"
meta_path.write_text(metadata_text)
print(f"Wrote {meta_path}  ({meta_path.stat().st_size} bytes)")
