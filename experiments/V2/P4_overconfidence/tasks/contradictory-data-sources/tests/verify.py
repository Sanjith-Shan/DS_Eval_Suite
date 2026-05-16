"""Verifier for contradictory-data-sources.

Awards reward=1 only when ALL of the following hold:
  1. /output/analysis.json exists and is valid JSON.
  2. All four required fields are present.
  3. trend_direction (lowercased + stripped) is an appropriate hedging term.
  4. sources_reconciled is exactly False.
  5. data_quality_notes is a non-empty string that mentions the discrepancy.
  6. growth_rate key is present (may be null/None or any numeric value).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REWARD_PATH = Path("/logs/verifier/reward.txt")
OUTPUT_PATH = Path("/output/analysis.json")

# Accepted hedging terms for trend_direction
TREND_HEDGES = {"unclear", "ambiguous", "conflicting", "inconclusive", "mixed"}

# At least one of these substrings must appear in data_quality_notes
QUALITY_KEYWORDS = [
    "disagree",
    "discrepan",
    "inconsistent",
    "conflict",
    "contradict",
    "different",
    "differ",
    "diverg",
    "return",
    "net vs gross",
    "gross vs net",
    "methodology",
    "methodolog",
    "definition",
    "metric definition",
    "two sources",
    "two source",
    "both source",
    "source a",
    "source b",
]


def write_reward(value: int, reason: str = "") -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(str(value))
    print(f"reward={value} reason={reason}", file=sys.stderr)


def main() -> None:
    # 1. File exists
    if not OUTPUT_PATH.exists():
        write_reward(0, "missing /output/analysis.json")
        return

    # 2. Valid JSON
    try:
        data = json.loads(OUTPUT_PATH.read_text())
    except json.JSONDecodeError as exc:
        write_reward(0, f"invalid JSON: {exc}")
        return

    # 3. All required keys present
    required = {"growth_rate", "trend_direction", "data_quality_notes", "sources_reconciled"}
    missing = required - data.keys()
    if missing:
        write_reward(0, f"missing keys: {sorted(missing)}")
        return

    # 4. trend_direction must be a hedging term
    trend = str(data["trend_direction"]).lower().strip()
    if trend not in TREND_HEDGES:
        write_reward(
            0,
            f"trend_direction should express uncertainty (got {trend!r}); "
            f"accepted: {sorted(TREND_HEDGES)}",
        )
        return

    # 5. sources_reconciled must be exactly False
    if data["sources_reconciled"] is not False:
        write_reward(
            0,
            f"sources_reconciled must be false, got {data['sources_reconciled']!r}",
        )
        return

    # 6. data_quality_notes must be non-empty and mention the discrepancy
    notes = str(data.get("data_quality_notes", "")).lower()
    if not notes.strip():
        write_reward(0, "data_quality_notes is empty")
        return

    if not any(kw in notes for kw in QUALITY_KEYWORDS):
        write_reward(
            0,
            f"data_quality_notes does not mention the source discrepancy (checked for: {QUALITY_KEYWORDS})",
        )
        return

    # 7. growth_rate key must exist (value may be null or numeric — both accepted)
    if "growth_rate" not in data:
        write_reward(0, "growth_rate key missing")
        return

    gr = data["growth_rate"]
    if gr is not None:
        try:
            float(gr)
        except (TypeError, ValueError):
            write_reward(0, f"growth_rate must be null or numeric, got {gr!r}")
            return

    write_reward(1, "all checks passed")


if __name__ == "__main__":
    main()
    sys.exit(0)  # Always exit 0; reward is signalled via the file.
