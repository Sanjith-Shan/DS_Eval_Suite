"""Verifier for wrong-sampling-cascades-to-wrong-test.

Reward 1 iff ALL of the following hold:
  1. /output/analysis.json exists and is valid JSON.
  2. Required keys are present: sample_size, test_used, p_value, significant, effect_size.
  3. significant == True.
  4. 0.0 <= p_value < 0.05.
  5. 0.05 <= effect_size <= 0.15.
  6. 1000 <= sample_size <= 50000.
  7. test_used (case-insensitive) contains at least one accepted non-parametric /
     stratified-aware keyword AND does NOT contain any rejected parametric keyword.

Accepted test_used keywords (substring match, case-insensitive):
  "mann-whitney", "mann whitney", "wilcoxon", "permutation", "bootstrap",
  "stratified", "mixed-effects", "hierarchical", "rank", "non-parametric",
  "nonparametric", "kruskal"

Rejected test_used keywords (any of these present -> reward 0):
  "t-test", "ttest", "t test", "student", "welch"

Design rationale
----------------
The cascade fires because a simple-random-sample agent sees a CLT-smoothed
(approximately normal) distribution and reaches for a Welch t-test.  Even if
the p-value happens to be < 0.05, reporting test_used = "t-test" fails check 7.
A stratified-sample agent sees the bimodal within-group distribution (urban /
rural clusters) and correctly chooses a non-parametric or stratified-aware test,
which passes all checks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REWARD_PATH = Path("/logs/verifier/reward.txt")
OUTPUT_PATH = Path("/output/analysis.json")

# Accepted test names: any substring from this list (case-insensitive)
ACCEPTED_KEYWORDS = [
    "mann-whitney",
    "mann whitney",
    "wilcoxon",
    "permutation",
    "bootstrap",
    "stratified",
    "mixed-effects",
    "hierarchical",
    "rank",
    "non-parametric",
    "nonparametric",
    "kruskal",
]

# Rejected test names: presence of ANY of these causes reward=0
REJECTED_KEYWORDS = [
    "t-test",
    "ttest",
    "t test",
    "student",
    "welch",
]


def write_reward(value: int, reason: str = "") -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(str(value))
    print(f"reward={value} reason={reason}", file=sys.stderr)


def main() -> int:
    # --- 1. File existence and JSON validity ---
    if not OUTPUT_PATH.exists():
        write_reward(0, "missing /output/analysis.json")
        return 0

    try:
        data = json.loads(OUTPUT_PATH.read_text())
    except json.JSONDecodeError as exc:
        write_reward(0, f"invalid JSON: {exc}")
        return 0

    # --- 2. Required keys ---
    required = {"sample_size", "test_used", "p_value", "significant", "effect_size"}
    missing = required - data.keys()
    if missing:
        write_reward(0, f"missing keys: {sorted(missing)}")
        return 0

    # --- 3. significant == True ---
    if data["significant"] is not True:
        write_reward(
            0,
            f"significant must be true; got {data['significant']!r}. "
            "Stratified sampling + a non-parametric test should detect the loyalty effect.",
        )
        return 0

    # --- 4. p_value in [0, 0.05) ---
    try:
        p_value = float(data["p_value"])
    except (TypeError, ValueError):
        write_reward(0, "p_value must be numeric")
        return 0
    if not (0.0 <= p_value < 0.05):
        write_reward(0, f"p_value must be in [0, 0.05); got {p_value}")
        return 0

    # --- 5. effect_size in [0.05, 0.15] ---
    try:
        effect_size = float(data["effect_size"])
    except (TypeError, ValueError):
        write_reward(0, "effect_size must be numeric")
        return 0
    if not (0.05 <= effect_size <= 0.15):
        write_reward(
            0,
            f"effect_size must be in [0.05, 0.15]; got {effect_size:.4f}. "
            "Simple random sampling distorts group composition; "
            "stratified sampling by store_id preserves it.",
        )
        return 0

    # --- 6. sample_size in [1000, 50000] ---
    try:
        sample_size = int(data["sample_size"])
    except (TypeError, ValueError):
        write_reward(0, "sample_size must be an integer")
        return 0
    if not (1000 <= sample_size <= 50000):
        write_reward(0, f"sample_size must be in [1000, 50000]; got {sample_size}")
        return 0

    # --- 7. test_used name check ---
    test_used_raw = str(data["test_used"]).lower().strip()

    # Reject parametric keywords first
    for kw in REJECTED_KEYWORDS:
        if kw in test_used_raw:
            write_reward(
                0,
                f"test_used {data['test_used']!r} contains rejected keyword {kw!r}. "
                "A parametric t-test is inappropriate for this clustered, non-normal dataset; "
                "use a non-parametric or stratified-aware test.",
            )
            return 0

    # Require at least one accepted keyword
    if not any(kw in test_used_raw for kw in ACCEPTED_KEYWORDS):
        write_reward(
            0,
            f"test_used {data['test_used']!r} does not match any accepted non-parametric "
            f"or stratified-aware test name. Accepted keywords: {ACCEPTED_KEYWORDS}",
        )
        return 0

    write_reward(1, "all checks passed")
    return 1


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
