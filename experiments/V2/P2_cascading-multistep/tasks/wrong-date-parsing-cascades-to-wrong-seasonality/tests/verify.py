"""Verifier for wrong-date-parsing-cascades-to-wrong-seasonality.

Reward 1 iff:
  - /output/forecast.json exists and is valid JSON
  - Required keys are present: peak_month, trough_month, forecast_next_3_months, seasonal_strength
  - peak_month == 12   (December is the correct seasonal peak)
  - trough_month == 6  (June is the correct seasonal trough)
  - seasonal_strength > 0.3
  - forecast_next_3_months is a list of exactly 3 floats, each between 400 and 1600

Note on seasonal_strength definition: the solution uses
  seasonal.std() / observed.std()
where 'observed' is the monthly sum series passed to seasonal_decompose.
This is a stable, model-free ratio that does not require separating trend
from residual variance.  Naive date parsing (dayfirst=False default) silently
swaps month and day for ~15% of EU-sourced rows, shifting them to wrong months
and causing the decomposition to report peak_month=1, trough_month=5, failing
both month checks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REWARD_PATH = Path("/logs/verifier/reward.txt")
OUTPUT_PATH = Path("/output/forecast.json")


def write_reward(value: int, reason: str = "") -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(str(value))
    print(f"reward={value} reason={reason}", file=sys.stderr)


def main() -> int:
    if not OUTPUT_PATH.exists():
        write_reward(0, "missing /output/forecast.json")
        return 0

    try:
        data = json.loads(OUTPUT_PATH.read_text())
    except json.JSONDecodeError as exc:
        write_reward(0, f"invalid JSON: {exc}")
        return 0

    required = {"peak_month", "trough_month", "forecast_next_3_months", "seasonal_strength"}
    missing = required - data.keys()
    if missing:
        write_reward(0, f"missing keys: {sorted(missing)}")
        return 0

    # --- peak_month ---
    try:
        peak_month = int(data["peak_month"])
    except (TypeError, ValueError):
        write_reward(0, f"peak_month must be an integer, got {data['peak_month']!r}")
        return 0
    if peak_month != 12:
        write_reward(0, f"peak_month should be 12 (December), got {peak_month}")
        return 0

    # --- trough_month ---
    try:
        trough_month = int(data["trough_month"])
    except (TypeError, ValueError):
        write_reward(0, f"trough_month must be an integer, got {data['trough_month']!r}")
        return 0
    if trough_month != 6:
        write_reward(0, f"trough_month should be 6 (June), got {trough_month}")
        return 0

    # --- seasonal_strength ---
    try:
        seasonal_strength = float(data["seasonal_strength"])
    except (TypeError, ValueError):
        write_reward(0, f"seasonal_strength must be a float, got {data['seasonal_strength']!r}")
        return 0
    if seasonal_strength <= 0.3:
        write_reward(0, f"seasonal_strength should be > 0.3, got {seasonal_strength:.4f}")
        return 0

    # --- forecast_next_3_months ---
    forecast = data["forecast_next_3_months"]
    if not isinstance(forecast, list) or len(forecast) != 3:
        write_reward(0, f"forecast_next_3_months must be a list of 3 elements, got {forecast!r}")
        return 0
    for i, val in enumerate(forecast):
        try:
            fval = float(val)
        except (TypeError, ValueError):
            write_reward(0, f"forecast_next_3_months[{i}] must be a float, got {val!r}")
            return 0
        if not (400.0 <= fval <= 1_600_000.0):
            write_reward(0, f"forecast_next_3_months[{i}]={fval} is outside plausible range [400, 1600000]")
            return 0

    write_reward(1, f"all checks passed — peak=12, trough=6, strength={seasonal_strength:.4f}")
    return 1


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
