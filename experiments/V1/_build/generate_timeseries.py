"""Generate a 3-year daily sales series with weekly and yearly seasonality,
plus a structural break at month 18 that permanently doubles the baseline.

The first 1095 days go into /workspace/sales.csv (visible to the agent).
The next 30 days (1096-1125) are written to /tests/holdout.csv for the
verifier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1] / "samples" / "time-series-regime-change"
WS_OUT = TASK_DIR / "environment" / "sales.csv"
HOLDOUT_OUT = TASK_DIR / "tests" / "holdout.csv"

SEED = 20260515
TRAIN_DAYS = 1095  # 3 years
HOLDOUT_DAYS = 30
BREAK_DAY = 540  # midway through year 2

BASELINE_PRE = 100.0
BASELINE_POST = 200.0
WEEKEND_FACTOR = 0.6
HOLIDAY_SPIKE = 1.35  # Nov-Dec uplift

NOISE_PRE = 10.0
NOISE_POST = 15.0


def generate(rng: np.random.Generator, n_days: int, start_offset: int = 0) -> pd.DataFrame:
    start_date = pd.Timestamp("2022-01-01") + pd.Timedelta(days=start_offset)
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    day_index = np.arange(n_days) + start_offset
    is_post = day_index >= BREAK_DAY
    baseline = np.where(is_post, BASELINE_POST, BASELINE_PRE)
    noise_std = np.where(is_post, NOISE_POST, NOISE_PRE)
    noise = rng.normal(0, 1, size=n_days) * noise_std

    weekday = pd.Series(dates).dt.dayofweek.values  # Mon=0..Sun=6
    weekend = (weekday >= 5).astype(float)
    weekday_factor = 1.0 * (1 - weekend) + WEEKEND_FACTOR * weekend

    month = pd.Series(dates).dt.month.values
    holiday_factor = np.where(np.isin(month, [11, 12]), HOLIDAY_SPIKE, 1.0)

    sales = baseline * weekday_factor * holiday_factor + noise
    sales = np.clip(sales, 0, None).round(2)

    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "sales": sales})


def main() -> None:
    rng = np.random.default_rng(SEED)
    full = generate(rng, TRAIN_DAYS + HOLDOUT_DAYS)
    train = full.iloc[:TRAIN_DAYS].copy()
    holdout = full.iloc[TRAIN_DAYS:].copy()

    WS_OUT.parent.mkdir(parents=True, exist_ok=True)
    HOLDOUT_OUT.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(WS_OUT, index=False)
    holdout.to_csv(HOLDOUT_OUT, index=False)
    print(f"wrote {WS_OUT} ({len(train)} rows)")
    print(f"wrote {HOLDOUT_OUT} ({len(holdout)} rows)")
    print(f"holdout mean={holdout['sales'].mean():.2f}  pre-break mean={train.iloc[:BREAK_DAY]['sales'].mean():.2f}  post-break mean={train.iloc[BREAK_DAY:]['sales'].mean():.2f}")


if __name__ == "__main__":
    main()
