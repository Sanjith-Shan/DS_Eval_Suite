"""Verifier for wrong-join-cascades-to-wrong-report.

Reward 1 iff:
  - /output/report.json exists and contains valid JSON
  - Keys: quarterly_revenue, underperforming_quarters, total_annual_revenue
  - total_annual_revenue is within 5% of $8,725,677 (true value with LEFT join)
    i.e. in the range [8,289,393, 9,161,961]
  - quarterly_revenue is a dict with Q1/Q2/Q3/Q4 keys and numeric values
  - "Q4" does NOT appear in underperforming_quarters (Q4 is not underperforming)
  - underperforming_quarters has at most 1 entry (none should be underperforming in truth)

INNER-join path:
  - Produces total_annual_revenue ~ $6,524,747 (fails the revenue check)
  - Places Q4 in underperforming_quarters (fails the Q4 check)
  Both checks fail, producing reward 0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REWARD_PATH = Path("/logs/verifier/reward.txt")
OUTPUT_PATH = Path("/output/report.json")

# True annual revenue (LEFT join, all 10 000 transactions)
TRUE_ANNUAL = 8_725_677.0
TOLERANCE   = 0.05   # 5%
ANNUAL_LO   = TRUE_ANNUAL * (1 - TOLERANCE)
ANNUAL_HI   = TRUE_ANNUAL * (1 + TOLERANCE)


def write_reward(value: int, reason: str = "") -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(str(value))
    print(f"reward={value} reason={reason}", file=sys.stderr)


def main() -> int:
    # --- existence ---
    if not OUTPUT_PATH.exists():
        write_reward(0, "missing /output/report.json")
        return 0

    # --- JSON validity ---
    try:
        data = json.loads(OUTPUT_PATH.read_text())
    except json.JSONDecodeError as exc:
        write_reward(0, f"invalid JSON: {exc}")
        return 0

    # --- required keys ---
    required = {"quarterly_revenue", "underperforming_quarters", "total_annual_revenue"}
    missing  = required - data.keys()
    if missing:
        write_reward(0, f"missing keys: {sorted(missing)}")
        return 0

    # --- total_annual_revenue numeric and in range ---
    try:
        total = float(data["total_annual_revenue"])
    except (TypeError, ValueError):
        write_reward(0, "total_annual_revenue must be numeric")
        return 0

    if not (ANNUAL_LO <= total <= ANNUAL_HI):
        write_reward(
            0,
            f"total_annual_revenue {total:,.0f} not within 5% of true value "
            f"{TRUE_ANNUAL:,.0f} (expected [{ANNUAL_LO:,.0f}, {ANNUAL_HI:,.0f}]). "
            f"Hint: INNER join drops orphan transactions and gives ~$6,524,747."
        )
        return 0

    # --- quarterly_revenue structure ---
    qrev = data["quarterly_revenue"]
    if not isinstance(qrev, dict):
        write_reward(0, "quarterly_revenue must be a dict")
        return 0

    expected_keys = {"Q1", "Q2", "Q3", "Q4"}
    missing_keys  = expected_keys - qrev.keys()
    if missing_keys:
        write_reward(0, f"quarterly_revenue missing quarters: {sorted(missing_keys)}")
        return 0

    for q in expected_keys:
        try:
            float(qrev[q])
        except (TypeError, ValueError):
            write_reward(0, f"quarterly_revenue[{q!r}] must be numeric")
            return 0

    # --- underperforming_quarters: Q4 must NOT be in it ---
    upq = data["underperforming_quarters"]
    if not isinstance(upq, list):
        write_reward(0, "underperforming_quarters must be a list")
        return 0

    # Normalise to upper-case for comparison
    upq_upper = [str(x).strip().upper() for x in upq]
    if "Q4" in upq_upper:
        write_reward(
            0,
            f"Q4 incorrectly flagged as underperforming: {upq}. "
            f"Q4 is NOT underperforming when all transactions are included via LEFT join."
        )
        return 0

    # Allow at most 1 entry (the correct answer has none; tolerate rounding edge cases)
    if len(upq) > 1:
        write_reward(
            0,
            f"underperforming_quarters has {len(upq)} entries {upq}; "
            f"expected at most 1 (and 0 for a perfect solution)."
        )
        return 0

    write_reward(1, "all checks passed")
    return 1


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
