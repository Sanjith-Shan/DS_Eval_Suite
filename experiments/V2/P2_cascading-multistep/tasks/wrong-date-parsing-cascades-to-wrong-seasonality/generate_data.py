"""Generate sales.csv for wrong-date-parsing-cascades-to-wrong-seasonality task.

Produces 3 years of daily sales (2022-01-01 to 2024-12-31) with a strong
December peak and June trough.  85% of rows use MM/DD/YYYY (US format),
15% use DD/MM/YYYY (EU format).  All EU rows have day <= 12, so pandas
default parsing silently misreads them (e.g., July 3 -> March 7).

Run this script once to regenerate environment/sales.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose


np.random.seed(42)

# ------------------------------------------------------------------
# 1.  Build the full date range and true seasonal sales
# ------------------------------------------------------------------
dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
n = len(dates)
print(f"Total rows: {n}")

base = 1000.0
amplitude = 400.0
months = dates.month.values  # 1-12

# Strong sinusoid: peak at December (month 12), trough at June (month 6)
seasonal_signal = base + amplitude * np.cos(2 * np.pi * (months - 12) / 12)
noise = np.random.normal(0, 30, n)
sales = seasonal_signal + noise

# ------------------------------------------------------------------
# 2.  Assign source (US / EU) — ALL EU rows must have day <= 12
# ------------------------------------------------------------------
days = dates.day.values
eligible_eu = np.where(days <= 12)[0]   # indices where day is ambiguous
n_eu_target = int(round(0.15 * n))

# Sample EU rows only from day-<=12 pool
eu_indices = np.random.choice(eligible_eu, size=min(n_eu_target, len(eligible_eu)), replace=False)
eu_set = set(eu_indices.tolist())

source = np.where([i in eu_set for i in range(n)], "EU", "US")
n_us = int((source == "US").sum())
n_eu = int((source == "EU").sum())
n_eu_day_le12 = int(sum(1 for i in eu_set if days[i] <= 12))

print(f"US rows: {n_us}")
print(f"EU rows: {n_eu}")
print(f"EU rows with day <= 12 (silently misparsed): {n_eu_day_le12}")

# ------------------------------------------------------------------
# 3.  Build date strings
#     US: MM/DD/YYYY  |  EU: DD/MM/YYYY (swaps month and day)
# ------------------------------------------------------------------
date_strings = []
for i, d in enumerate(dates):
    if source[i] == "US":
        date_strings.append(d.strftime("%m/%d/%Y"))
    else:
        # EU format: DD/MM/YYYY  (e.g., July 3 -> "03/07/2022")
        date_strings.append(d.strftime("%d/%m/%Y"))

df = pd.DataFrame({"date": date_strings, "sales": sales.round(2), "source": source})


# ------------------------------------------------------------------
# 4.  Demonstrate the cascade: naive parse vs correct parse
# ------------------------------------------------------------------
def monthly_decomp(daily_series: pd.Series) -> dict:
    """Return peak_month, trough_month, seasonal_strength from a daily Series."""
    monthly = daily_series.resample("MS").sum()
    # Need at least 2 full periods for decomposition (period=12 -> 24 months)
    result = seasonal_decompose(monthly, model="additive", period=12, extrapolate_trend="freq")
    seasonal = result.seasonal
    observed = result.observed
    peak_month = int(seasonal.groupby(seasonal.index.month).mean().idxmax())
    trough_month = int(seasonal.groupby(seasonal.index.month).mean().idxmin())
    strength = float(seasonal.std() / observed.std())
    return {"peak_month": peak_month, "trough_month": trough_month, "seasonal_strength": round(strength, 4)}


print("\n--- NAIVE PARSE (default pd.to_datetime, dayfirst=False) ---")
naive_dates = pd.to_datetime(df["date"], dayfirst=False, errors="coerce")
naive_series = pd.Series(df["sales"].values, index=naive_dates, name="sales").dropna()
naive_series = naive_series.sort_index()
naive = monthly_decomp(naive_series)
print(f"  peak_month:        {naive['peak_month']}")
print(f"  trough_month:      {naive['trough_month']}")
print(f"  seasonal_strength: {naive['seasonal_strength']}")

print("\n--- CORRECT PARSE (split by source column) ---")
us_mask = df["source"] == "US"
eu_mask = df["source"] == "EU"
us_dates = pd.to_datetime(df.loc[us_mask, "date"], format="%m/%d/%Y")
eu_dates = pd.to_datetime(df.loc[eu_mask, "date"], format="%d/%m/%Y")
correct_dates = pd.concat([us_dates, eu_dates]).sort_index()
correct_sales = pd.Series(
    df["sales"].values,
    index=pd.concat([
        pd.Series(us_dates.values, index=df.index[us_mask]),
        pd.Series(eu_dates.values, index=df.index[eu_mask]),
    ]).sort_index().values,
    name="sales",
).sort_index()
correct = monthly_decomp(correct_sales)
print(f"  peak_month:        {correct['peak_month']}")
print(f"  trough_month:      {correct['trough_month']}")
print(f"  seasonal_strength: {correct['seasonal_strength']}")

# ------------------------------------------------------------------
# 5.  Sanity assertions
# ------------------------------------------------------------------
assert correct["peak_month"] == 12, f"Expected correct peak=12, got {correct['peak_month']}"
assert correct["trough_month"] == 6, f"Expected correct trough=6, got {correct['trough_month']}"
assert correct["seasonal_strength"] > 0.4, f"Expected correct strength>0.4, got {correct['seasonal_strength']}"
# Naive parse should produce wrong peak or wrong trough (the cascade)
assert naive["peak_month"] != 12 or naive["trough_month"] != 6, (
    "Naive parse should produce wrong peak or trough — cascade is too weak"
)
print(f"\nCascade confirmed: naive gives peak={naive['peak_month']}, trough={naive['trough_month']} "
      f"(correct: peak=12, trough=6)")

print("\nAll assertions passed.")

# ------------------------------------------------------------------
# 6.  Write CSV
# ------------------------------------------------------------------
out_path = Path(__file__).parent / "environment" / "sales.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"\nWrote {len(df)} rows to {out_path}")
