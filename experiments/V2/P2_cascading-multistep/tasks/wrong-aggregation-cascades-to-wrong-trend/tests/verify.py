"""Verifier for wrong-aggregation-cascades-to-wrong-trend.

Reward 1 iff:
  - /output/analysis.json exists and is valid JSON
  - Required keys are present: daily_summary, anomalous_days, anomaly_explanation
  - daily_summary contains at least 85 date entries with numeric values
  - anomalous_days is a list with 2–5 entries (correct mean-based answer: 3 planted days)
  - At least 2 of the 3 known planted anomaly dates appear in anomalous_days

A sum-based wrong answer flags ~9–12 days (weekends + planted), exceeding the length limit
and therefore receives reward 0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REWARD_PATH = Path("/logs/verifier/reward.txt")
OUTPUT_PATH = Path("/output/analysis.json")

# Ground-truth planted anomaly dates (known to verifier, not in environment/)
PLANTED_ANOMALY_DATES = {"2024-01-16", "2024-02-17", "2024-03-13"}


def write_reward(value: int, reason: str = "") -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(str(value))
    print(f"reward={value} reason={reason}", file=sys.stderr)


def main() -> int:
    # ── 1. File existence ────────────────────────────────────────────────────
    if not OUTPUT_PATH.exists():
        write_reward(0, "missing /output/analysis.json")
        return 0

    # ── 2. Valid JSON ─────────────────────────────────────────────────────────
    try:
        data = json.loads(OUTPUT_PATH.read_text())
    except json.JSONDecodeError as exc:
        write_reward(0, f"invalid JSON: {exc}")
        return 0

    # ── 3. Required keys ──────────────────────────────────────────────────────
    required = {"daily_summary", "anomalous_days", "anomaly_explanation"}
    missing = required - data.keys()
    if missing:
        write_reward(0, f"missing keys: {sorted(missing)}")
        return 0

    # ── 4. daily_summary validation ───────────────────────────────────────────
    daily_summary = data["daily_summary"]
    if not isinstance(daily_summary, dict):
        write_reward(0, "daily_summary must be a dict")
        return 0

    if len(daily_summary) < 85:
        write_reward(0, f"daily_summary has only {len(daily_summary)} entries; expected >= 85")
        return 0

    for date_key, val in daily_summary.items():
        try:
            float(val)
        except (TypeError, ValueError):
            write_reward(0, f"daily_summary[{date_key!r}] is not numeric: {val!r}")
            return 0

    # ── 5. anomalous_days length check ────────────────────────────────────────
    anomalous_days = data["anomalous_days"]
    if not isinstance(anomalous_days, list):
        write_reward(0, "anomalous_days must be a list")
        return 0

    n_flagged = len(anomalous_days)
    if n_flagged < 2:
        write_reward(0, f"anomalous_days has {n_flagged} entries; too few (expected 2–5)")
        return 0

    if n_flagged > 5:
        write_reward(
            0,
            f"anomalous_days has {n_flagged} entries; expected 2–5. "
            "A daily-sum approach mislabels weekends as anomalous due to fewer operating hours "
            "on weekends. Use daily mean instead.",
        )
        return 0

    # ── 6. Planted anomaly recall ─────────────────────────────────────────────
    flagged_set = set(str(d).strip() for d in anomalous_days)
    found_planted = PLANTED_ANOMALY_DATES & flagged_set
    if len(found_planted) < 2:
        write_reward(
            0,
            f"only {len(found_planted)} of 3 planted anomaly dates found in anomalous_days "
            f"(found: {sorted(found_planted)}, expected at least 2 of {sorted(PLANTED_ANOMALY_DATES)})",
        )
        return 0

    write_reward(1, f"all checks passed — {n_flagged} anomalous days detected, {len(found_planted)} planted dates confirmed")
    return 1


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
