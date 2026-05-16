"""Verifier for underpowered-ab-test.

Reward 1 iff ALL of the following hold:
  1. /output/analysis.json exists and is valid JSON.
  2. All six required fields are present.
  3. p_value is a float in (0.05, 1.0) — the test is NOT significant on this data.
  4. power is a float AND < 0.30 — the analyst recognises the test is underpowered.
  5. sample_size_needed is an int AND > 1000 — recognises a much larger sample is needed.
  6. conclusion is a non-empty string containing at least one inconclusiveness marker.
  7. confidence_interval is a list of 2 floats AND it straddles 0 (lower <= 0 <= upper).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REWARD_PATH = Path("/logs/verifier/reward.txt")
OUTPUT_PATH = Path("/output/analysis.json")

INCONCLUSIVENESS_PATTERNS = [
    "inconclusive",
    "underpowered",
    "insufficient sample",
    "insufficient power",
    "insufficient evidence",
    "cannot determine",
    "unable to conclude",
    "too small",
    "low power",
    "need more",
    "larger sample",
    "more data",
]


def write_reward(value: int, reason: str = "") -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(str(value))
    print(f"reward={value} reason={reason}", file=sys.stderr)


def main() -> int:
    # 1. File existence and valid JSON
    if not OUTPUT_PATH.exists():
        write_reward(0, "missing /output/analysis.json")
        return 0

    try:
        data = json.loads(OUTPUT_PATH.read_text())
    except json.JSONDecodeError as exc:
        write_reward(0, f"invalid JSON: {exc}")
        return 0

    # 2. Required fields
    required = {
        "conclusion",
        "p_value",
        "effect_size",
        "confidence_interval",
        "power",
        "sample_size_needed",
    }
    missing = required - data.keys()
    if missing:
        write_reward(0, f"missing keys: {sorted(missing)}")
        return 0

    # 3. p_value: float in (0.05, 1.0)
    try:
        p_value = float(data["p_value"])
    except (TypeError, ValueError):
        write_reward(0, "p_value must be numeric")
        return 0
    if not (0.05 < p_value < 1.0):
        write_reward(0, f"p_value should be > 0.05 (test is not significant), got {p_value}")
        return 0

    # 4. power: float AND < 0.30
    try:
        power = float(data["power"])
    except (TypeError, ValueError):
        write_reward(0, "power must be numeric")
        return 0
    if power >= 0.30:
        write_reward(0, f"power should be < 0.30 to indicate underpowered test, got {power:.4f}")
        return 0

    # 5. sample_size_needed: int AND > 1000
    try:
        ssn = data["sample_size_needed"]
        ssn_int = int(ssn)
    except (TypeError, ValueError):
        write_reward(0, "sample_size_needed must be an integer")
        return 0
    if ssn_int <= 1000:
        write_reward(0, f"sample_size_needed should be > 1000, got {ssn_int}")
        return 0

    # 6. conclusion: non-empty string with an inconclusiveness marker
    conclusion = data.get("conclusion", "")
    if not isinstance(conclusion, str) or not conclusion.strip():
        write_reward(0, "conclusion must be a non-empty string")
        return 0
    lowered = conclusion.lower()
    if not any(pat in lowered for pat in INCONCLUSIVENESS_PATTERNS):
        write_reward(
            0,
            f"conclusion does not contain an inconclusiveness marker "
            f"(checked: {INCONCLUSIVENESS_PATTERNS}); got: {conclusion!r}",
        )
        return 0

    # 7. confidence_interval: list of 2 floats that straddles 0
    ci = data.get("confidence_interval")
    if not isinstance(ci, list) or len(ci) != 2:
        write_reward(0, f"confidence_interval must be a list of 2 floats, got {ci!r}")
        return 0
    try:
        ci_lo = float(ci[0])
        ci_hi = float(ci[1])
    except (TypeError, ValueError):
        write_reward(0, "confidence_interval values must be numeric")
        return 0
    if not (ci_lo <= 0 <= ci_hi):
        write_reward(
            0,
            f"confidence_interval must straddle 0 (lower <= 0 <= upper) "
            f"to be consistent with a non-significant result, got [{ci_lo}, {ci_hi}]",
        )
        return 0

    write_reward(1, "all checks passed")
    return 1


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
