"""Reference solution for wrong-date-parsing-cascades-to-wrong-seasonality.

Key insight: the CSV has a 'source' column with values "US" and "EU".
US dates are in MM/DD/YYYY format; EU dates are in DD/MM/YYYY format.
Parsing with explicit format strings per source avoids the silent
misparse that pandas default (dayfirst=False) would apply to the EU rows.

Steps:
  1. Parse US and EU dates separately with explicit format strings.
  2. Aggregate daily sales to monthly sums.
  3. Fit additive seasonal decomposition (period=12).
  4. Extract peak/trough from the average seasonal component by month.
  5. Compute seasonal_strength = seasonal.std() / observed.std().
  6. Forecast Jan/Feb/Mar 2025 as last_trend_value + mean_seasonal[month].
  7. Write /output/forecast.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


SALES_PATH = os.environ.get("SALES_PATH", "/workspace/sales.csv")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/forecast.json")


def main() -> None:
    df = pd.read_csv(SALES_PATH)

    # Parse dates by source to avoid silent misparse of EU DD/MM/YYYY rows.
    us_mask = df["source"] == "US"
    eu_mask = df["source"] == "EU"

    us_dates = pd.to_datetime(df.loc[us_mask, "date"], format="%m/%d/%Y")
    eu_dates = pd.to_datetime(df.loc[eu_mask, "date"], format="%d/%m/%Y")

    # Reconstruct a single Series indexed by the correctly parsed dates.
    date_index = pd.Series(index=df.index, dtype="object")
    date_index[us_mask] = us_dates.values
    date_index[eu_mask] = eu_dates.values
    date_index = pd.to_datetime(date_index)

    sales = pd.Series(df["sales"].values, index=date_index, name="sales")
    sales = sales.sort_index()

    # Monthly aggregation (sum of daily sales).
    monthly = sales.resample("MS").sum()

    # Additive seasonal decomposition with period=12 months.
    result = seasonal_decompose(monthly, model="additive", period=12, extrapolate_trend="freq")

    seasonal = result.seasonal
    trend = result.trend
    observed = result.observed

    # Average seasonal component by calendar month.
    avg_seasonal = seasonal.groupby(seasonal.index.month).mean()

    peak_month = int(avg_seasonal.idxmax())
    trough_month = int(avg_seasonal.idxmin())

    # seasonal_strength: seasonal component std / observed series std.
    seasonal_strength = float(seasonal.std() / observed.std())

    # Forecast Jan/Feb/Mar 2025: last available trend + average seasonal effect.
    last_trend = float(trend.iloc[-1])
    forecast_months = [1, 2, 3]  # Jan, Feb, Mar 2025
    forecast_next_3_months = [
        round(last_trend + float(avg_seasonal.loc[m]), 2)
        for m in forecast_months
    ]

    out = {
        "peak_month": peak_month,
        "trough_month": trough_month,
        "forecast_next_3_months": forecast_next_3_months,
        "seasonal_strength": round(seasonal_strength, 4),
    }

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_PATH).write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUTPUT_PATH}")
    print(f"  peak_month={peak_month}, trough_month={trough_month}, "
          f"seasonal_strength={seasonal_strength:.4f}")
    print(f"  forecast_next_3_months={forecast_next_3_months}")


if __name__ == "__main__":
    main()
