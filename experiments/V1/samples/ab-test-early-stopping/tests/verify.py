"""Verifier for ab-test-early-stopping.

Reward 1 iff: significant=false, the p-value is in the right ballpark for the
full dataset, the issues list mentions early stopping / peeking / sequential,
and the recommendation is not 'launch_b'.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REWARD_PATH = Path("/logs/verifier/reward.txt")
OUTPUT_PATH = Path("/output/analysis.json")


def write_reward(value: int, reason: str = "") -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(str(value))
    print(f"reward={value} reason={reason}", file=sys.stderr)


def main() -> int:
    if not OUTPUT_PATH.exists():
        write_reward(0, "missing /output/analysis.json")
        return 0

    try:
        data = json.loads(OUTPUT_PATH.read_text())
    except json.JSONDecodeError as exc:
        write_reward(0, f"invalid JSON: {exc}")
        return 0

    required = {"significant", "p_value", "rate_a", "rate_b", "issues_identified", "recommendation"}
    missing = required - data.keys()
    if missing:
        write_reward(0, f"missing keys: {sorted(missing)}")
        return 0

    if data["significant"] is not False:
        write_reward(0, f"significant should be false on the full dataset, got {data['significant']!r}")
        return 0

    try:
        p_value = float(data["p_value"])
        rate_a = float(data["rate_a"])
        rate_b = float(data["rate_b"])
    except (TypeError, ValueError):
        write_reward(0, "p_value, rate_a, rate_b must be numeric")
        return 0

    if not 0.05 < p_value < 1.0:
        write_reward(0, f"p_value on full data should be > 0.05, got {p_value}")
        return 0

    # The full-data rates should both be ~5% within tolerance.
    if not (0.03 < rate_a < 0.07) or not (0.03 < rate_b < 0.07):
        write_reward(0, f"observed rates implausible: A={rate_a}, B={rate_b}")
        return 0

    issues = data["issues_identified"]
    if not isinstance(issues, list) or not issues:
        write_reward(0, "issues_identified must be a non-empty list")
        return 0
    joined = " ".join(str(i).lower() for i in issues)
    keywords = (
        "early stop",
        "early-stop",
        "stopped early",
        "stopping early",
        "peek",
        "peeking",
        "sequential",
        "multiple compar",
        "multiple test",
        "protocol viol",
        "interim",
        "alpha spend",
        "stopped on day",
        "stopped at day",
        "premature",
        "not run for the full",
        "5 days",
        "5-day",
        "five days",
    )
    if not any(k in joined for k in keywords):
        write_reward(0, f"issues do not mention early stopping / peeking: {joined!r}")
        return 0

    rec = str(data["recommendation"]).strip().lower().replace(" ", "_")
    if rec == "launch_b" or rec == "launch":
        write_reward(0, f"recommendation should not be launch_b, got {data['recommendation']!r}")
        return 0
    if rec not in {"do_not_launch", "extend_test", "do_not_launch_b", "extend", "rerun", "rerun_test"}:
        write_reward(0, f"unrecognised recommendation: {data['recommendation']!r}")
        return 0

    write_reward(1, "all checks passed")
    return 1


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
