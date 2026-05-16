"""Verifier for simpsons-paradox.

Reward 1 iff:
  - better_treatment == "A"
  - aggregate / stratified rates within tolerance of ground truth
  - stratified_analysis == true
  - paradox_identified == true
  - explanation mentions Simpson's paradox or severity-as-confounder
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


REWARD_PATH = Path("/logs/verifier/reward.txt")
OUTPUT_PATH = Path("/output/analysis.json")

TRUE = {
    "agg_a": 273 / 350,
    "agg_b": 289 / 350,
    "mild_a": 81 / 87,
    "mild_b": 234 / 270,
    "severe_a": 192 / 263,
    "severe_b": 55 / 80,
}
TOL = 0.01


def write_reward(value: int, reason: str = "") -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(str(value))
    print(f"reward={value} reason={reason}", file=sys.stderr)


def close_enough(value, target, tol=TOL) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return abs(v - target) <= tol


def main() -> int:
    if not OUTPUT_PATH.exists():
        write_reward(0, "missing /output/analysis.json")
        return 0

    try:
        data = json.loads(OUTPUT_PATH.read_text())
    except json.JSONDecodeError as exc:
        write_reward(0, f"invalid JSON: {exc}")
        return 0

    required = {
        "better_treatment",
        "aggregate_rate_a",
        "aggregate_rate_b",
        "stratified_rates",
        "stratified_analysis",
        "paradox_identified",
        "explanation",
    }
    missing = required - data.keys()
    if missing:
        write_reward(0, f"missing keys: {sorted(missing)}")
        return 0

    if str(data["better_treatment"]).strip().upper() != "A":
        write_reward(0, f"better_treatment must be 'A', got {data['better_treatment']!r}")
        return 0

    if not close_enough(data["aggregate_rate_a"], TRUE["agg_a"]):
        write_reward(0, f"aggregate_rate_a off: got {data['aggregate_rate_a']}, expected ~{TRUE['agg_a']:.4f}")
        return 0
    if not close_enough(data["aggregate_rate_b"], TRUE["agg_b"]):
        write_reward(0, f"aggregate_rate_b off: got {data['aggregate_rate_b']}, expected ~{TRUE['agg_b']:.4f}")
        return 0

    strat = data["stratified_rates"]
    if not isinstance(strat, dict):
        write_reward(0, "stratified_rates must be an object")
        return 0
    for severity, prefix in [("mild", "mild"), ("severe", "severe")]:
        if severity not in strat:
            write_reward(0, f"stratified_rates missing {severity!r}")
            return 0
        sub = strat[severity]
        if not isinstance(sub, dict) or "A" not in sub or "B" not in sub:
            write_reward(0, f"stratified_rates[{severity!r}] must contain A and B")
            return 0
        if not close_enough(sub["A"], TRUE[f"{prefix}_a"]):
            write_reward(0, f"stratified_rates[{severity}][A] off")
            return 0
        if not close_enough(sub["B"], TRUE[f"{prefix}_b"]):
            write_reward(0, f"stratified_rates[{severity}][B] off")
            return 0

    if data["stratified_analysis"] is not True:
        write_reward(0, "stratified_analysis must be true")
        return 0
    if data["paradox_identified"] is not True:
        write_reward(0, "paradox_identified must be true")
        return 0

    explanation = str(data["explanation"]).lower()
    keywords = (
        "simpson",
        "lurking",
        "confound",
        "severity",
        "subgroup",
        "stratif",
        "aggregat",
        "allocation",
    )
    if not any(k in explanation for k in keywords):
        write_reward(0, f"explanation does not reference paradox / confounder: {explanation!r}")
        return 0

    write_reward(1, "all checks passed")
    return 1


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
