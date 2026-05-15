"""Verifier for time-series-regime-change.

Checks the agent's forecast against the held-out next 30 days. Reward 1 iff
MAPE < 20% AND the mean predicted level is in the post-regime range (>150)
AND no negative forecasts AND the dates align with the holdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REWARD_PATH = Path("/logs/verifier/reward.txt")
FORECAST_PATH = Path("/output/forecast.csv")
HOLDOUT_PATH = Path("/tests/holdout.csv")


def write_reward(value: int, reason: str = "") -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(str(value))
    print(f"reward={value} reason={reason}", file=sys.stderr)


def main() -> int:
    if not FORECAST_PATH.exists():
        write_reward(0, "missing /output/forecast.csv")
        return 0
    if not HOLDOUT_PATH.exists():
        write_reward(0, "/tests/holdout.csv missing (internal verifier error)")
        return 0

    try:
        fc = pd.read_csv(FORECAST_PATH)
    except Exception as exc:
        write_reward(0, f"could not read forecast CSV: {exc}")
        return 0

    if list(fc.columns) != ["date", "predicted_sales"]:
        write_reward(0, f"columns wrong; got {list(fc.columns)}")
        return 0
    if len(fc) != 30:
        write_reward(0, f"expected 30 rows, got {len(fc)}")
        return 0

    try:
        pred = pd.to_numeric(fc["predicted_sales"], errors="raise")
    except Exception as exc:
        write_reward(0, f"predicted_sales not numeric: {exc}")
        return 0
    if (pred < 0).any():
        write_reward(0, "negative forecast values detected")
        return 0
    if pred.mean() <= 150:
        write_reward(0, f"forecast mean {pred.mean():.2f} is below 150 (regime change not detected)")
        return 0

    holdout = pd.read_csv(HOLDOUT_PATH)

    try:
        fc_dates = pd.to_datetime(fc["date"]).dt.strftime("%Y-%m-%d").tolist()
    except Exception as exc:
        write_reward(0, f"date parsing failed: {exc}")
        return 0
    holdout_dates = holdout["date"].tolist()
    if fc_dates != holdout_dates:
        write_reward(0, f"forecast dates do not match holdout window; first 3 forecast={fc_dates[:3]}, holdout={holdout_dates[:3]}")
        return 0

    actual = holdout["sales"].values
    predicted = pred.values
    mape = float(np.mean(np.abs((actual - predicted) / np.where(actual == 0, 1, actual))) * 100)

    if mape >= 20.0:
        write_reward(0, f"MAPE {mape:.2f}% >= 20%")
        return 0

    write_reward(1, f"all checks passed; MAPE={mape:.2f}% mean_pred={pred.mean():.2f}")
    return 1


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
