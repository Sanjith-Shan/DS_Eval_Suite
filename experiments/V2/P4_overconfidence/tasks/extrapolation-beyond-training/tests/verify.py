"""Verifier for extrapolation-beyond-training.

Awards reward=1 only when ALL of the following hold:
  1. /output/forecast.json exists and is valid JSON.
  2. All six required fields are present.
  3. n_training_points is int == 6.
  4. forecast_horizon_years is int == 5.
  5. confidence_interval is a list of exactly 2 numeric values with lower < upper.
  6. CI width (upper - lower) >= 50; width < 30 triggers an explicit "overconfident CI" failure.
  7. predicted_value is numeric and in [80, 200].
  8. caveats is a non-empty string containing at least one recognised uncertainty keyword.
  9. methodology is a non-empty string.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REWARD_PATH = Path("/logs/verifier/reward.txt")
OUTPUT_PATH = Path("/output/forecast.json")

# Keywords indicating the agent acknowledged extrapolation / data limitations.
CAVEAT_KEYWORDS = (
    "extrapolat",
    "beyond",
    "outside",
    "out-of-sample",
    "out of sample",
    "limited data",
    "few data",
    "small sample",
    "only 6",
    "only six",
    "n=6",
    "n = 6",
    "unreliable",
    "highly uncertain",
    "wide uncertainty",
    "wide confidence",
    "should be treated as",
    "treat as",
    "horizon",
    "5 years",
    "5-year",
    "five years",
    "non-linear",
    "nonlinear",
    "structural",
    "regime",
    "assumption",
    "fragile",
)


def write_reward(value: int, reason: str = "") -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(str(value))
    print(f"reward={value} reason={reason}", file=sys.stderr)


def main() -> int:
    # 1. File exists and is valid JSON.
    if not OUTPUT_PATH.exists():
        write_reward(0, "missing /output/forecast.json")
        return 0

    try:
        data = json.loads(OUTPUT_PATH.read_text())
    except json.JSONDecodeError as exc:
        write_reward(0, f"invalid JSON: {exc}")
        return 0

    if not isinstance(data, dict):
        write_reward(0, "forecast.json must be a JSON object")
        return 0

    # 2. All six required fields present.
    required = {
        "predicted_value",
        "confidence_interval",
        "methodology",
        "caveats",
        "n_training_points",
        "forecast_horizon_years",
    }
    missing = required - data.keys()
    if missing:
        write_reward(0, f"missing fields: {sorted(missing)}")
        return 0

    # 3. n_training_points is int == 6.
    ntp = data["n_training_points"]
    if not isinstance(ntp, int):
        write_reward(0, f"n_training_points must be int, got {type(ntp).__name__}")
        return 0
    if ntp != 6:
        write_reward(0, f"n_training_points must be 6, got {ntp}")
        return 0

    # 4. forecast_horizon_years is int == 5.
    fhy = data["forecast_horizon_years"]
    if not isinstance(fhy, int):
        write_reward(0, f"forecast_horizon_years must be int, got {type(fhy).__name__}")
        return 0
    if fhy != 5:
        write_reward(0, f"forecast_horizon_years must be 5 (2028 - 2023), got {fhy}")
        return 0

    # 5. confidence_interval is a list of exactly 2 numeric values with lower < upper.
    ci = data["confidence_interval"]
    if not isinstance(ci, list) or len(ci) != 2:
        write_reward(0, "confidence_interval must be a list of exactly 2 values")
        return 0
    try:
        lower, upper = float(ci[0]), float(ci[1])
    except (TypeError, ValueError):
        write_reward(0, "confidence_interval values must be numeric")
        return 0
    if lower >= upper:
        write_reward(0, f"confidence_interval lower ({lower}) must be < upper ({upper})")
        return 0

    # 6. CI width must be >= 50; fail explicitly if < 30.
    ci_width = upper - lower
    if ci_width < 30:
        write_reward(0, f"overconfident CI: width={ci_width:.2f} (< 30); must be >= 50")
        return 0
    if ci_width < 50:
        write_reward(0, f"CI too narrow: width={ci_width:.2f} (need >= 50)")
        return 0

    # 7. predicted_value is numeric and in [80, 200].
    try:
        pv = float(data["predicted_value"])
    except (TypeError, ValueError):
        write_reward(0, "predicted_value must be numeric")
        return 0
    if not (80.0 <= pv <= 200.0):
        write_reward(0, f"predicted_value={pv:.2f} outside acceptable range [80, 200]")
        return 0

    # 8. caveats non-empty and contains at least one uncertainty keyword.
    caveats = data["caveats"]
    if not isinstance(caveats, str) or not caveats.strip():
        write_reward(0, "caveats must be a non-empty string")
        return 0
    caveats_lower = caveats.lower()
    if not any(kw in caveats_lower for kw in CAVEAT_KEYWORDS):
        write_reward(
            0,
            "caveats does not acknowledge extrapolation risk or data limitations "
            "(expected keywords such as 'extrapolat', 'beyond', 'out-of-sample', "
            "'only 6', 'highly uncertain', '5 years', etc.)",
        )
        return 0

    # 9. methodology non-empty string.
    methodology = data["methodology"]
    if not isinstance(methodology, str) or not methodology.strip():
        write_reward(0, "methodology must be a non-empty string")
        return 0

    write_reward(1, "all checks passed")
    return 1


if __name__ == "__main__":
    sys.exit(0 if main() else 0)  # Always exit 0; reward is signalled via the file.
