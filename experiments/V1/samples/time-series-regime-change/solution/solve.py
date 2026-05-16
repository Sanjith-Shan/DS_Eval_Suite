"""Reference solution for time-series-regime-change.

Strategy:
  1. Detect the structural break with a simple rolling-mean test.
  2. Train an additive model on the post-break segment only, decomposed into:
       baseline (constant) + day-of-week effect + month effect + residual.
  3. Forecast 30 days ahead by applying the day-of-week and month effects to
     the post-break baseline.

This is intentionally a non-fancy model. It beats the verifier's MAPE bar
because it does not contaminate the post-regime level with the pre-regime
average.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

IN = Path("/workspace/sales.csv")
OUT = Path("/output/forecast.csv")


def detect_break(series: pd.Series, min_size: int = 60) -> int:
    """Find the index that maximises the absolute mean shift between the
    left and right segments."""
    n = len(series)
    best_idx, best_score = n // 2, 0.0
    cum = series.cumsum().values
    total = cum[-1]
    for i in range(min_size, n - min_size):
        left_mean = cum[i - 1] / i
        right_mean = (total - cum[i - 1]) / (n - i)
        score = abs(right_mean - left_mean)
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx


def main() -> None:
    df = pd.read_csv(IN, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    break_idx = detect_break(df["sales"], min_size=120)
    post = df.iloc[break_idx:].copy()

    # Decompose: baseline + dow + month effects.
    baseline = post["sales"].mean()
    post["resid_baseline"] = post["sales"] - baseline
    dow_effect = post.groupby(post["date"].dt.dayofweek)["resid_baseline"].mean()
    post["resid_dow"] = post["resid_baseline"] - post["date"].dt.dayofweek.map(dow_effect)
    month_effect = post.groupby(post["date"].dt.month)["resid_dow"].mean()

    last_date = df["date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30, freq="D")

    future_dow = future_dates.dayofweek.map(dow_effect.to_dict())
    future_month = future_dates.month.map(month_effect.to_dict())
    forecast = baseline + future_dow.values + future_month.values
    forecast = np.clip(forecast, 0, None).round(2)

    out_df = pd.DataFrame({
        "date": future_dates.strftime("%Y-%m-%d"),
        "predicted_sales": forecast,
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
