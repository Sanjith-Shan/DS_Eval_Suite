"""Verifier for p3-survivorship-bias-in-dataset.

Reward 1 iff ALL of the following hold:
  1. /output/analysis.md exists.
  2. The markdown (lowercased) mentions at least 2 of the analysis-content keywords,
     proving the agent actually performed an analysis:
       employee, funding, marketing, industry, coefficient, p-value, significant,
       predictor, r-squared, r², regression
  3. The markdown (lowercased) contains at least one bias-awareness keyword,
     proving the agent flagged the data quality issue disclosed in the README:
       survivorship, survival bias, selection bias, failed compan, failed business,
       only active, only surviving, censoring, censored, non-representative,
       generaliz, sampling bias, sample selection
"""

from __future__ import annotations

import sys
from pathlib import Path


REWARD_PATH = Path("/logs/verifier/reward.txt")
LOG_PATH = Path("/logs/verifier/output.log")
OUTPUT_PATH = Path("/output/analysis.md")

ANALYSIS_KEYWORDS = [
    "employee",
    "funding",
    "marketing",
    "industry",
    "coefficient",
    "p-value",
    "significant",
    "predictor",
    "r-squared",
    "r²",
    "regression",
]

BIAS_KEYWORDS = [
    "survivorship",
    "survival bias",
    "selection bias",
    "failed compan",
    "failed business",
    "only active",
    "only surviving",
    "censoring",
    "censored",
    "non-representative",
    "generaliz",
    "sampling bias",
    "sample selection",
]


def write_reward(value: int, reason: str = "") -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(str(value))
    LOG_PATH.write_text(f"reward={value} reason={reason}\n")
    print(f"reward={value} reason={reason}", file=sys.stderr)


def main() -> int:
    # ------------------------------------------------------------------
    # 1. Check file existence
    # ------------------------------------------------------------------
    if not OUTPUT_PATH.exists():
        write_reward(0, "missing /output/analysis.md")
        return 0

    content = OUTPUT_PATH.read_text().lower()

    # ------------------------------------------------------------------
    # 2. Analysis content check — at least 2 keywords must appear
    # ------------------------------------------------------------------
    matched_analysis = [kw for kw in ANALYSIS_KEYWORDS if kw in content]
    if len(matched_analysis) < 2:
        write_reward(
            0,
            f"agent did not perform an analysis — fewer than 2 analysis keywords "
            f"found. Matched: {matched_analysis}",
        )
        return 0

    # ------------------------------------------------------------------
    # 3. Bias awareness check — at least 1 keyword must appear
    # ------------------------------------------------------------------
    matched_bias = [kw for kw in BIAS_KEYWORDS if kw in content]
    if not matched_bias:
        write_reward(
            0,
            "agent failed to flag the survivorship bias from the README — "
            "none of the bias-awareness keywords found in report.",
        )
        return 0

    # ------------------------------------------------------------------
    # All checks passed
    # ------------------------------------------------------------------
    write_reward(
        1,
        f"all checks passed (analysis keywords: {matched_analysis}; "
        f"bias keywords: {matched_bias})",
    )
    return 1


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
