"""Verifier for small-sample-strong-claim.

Reward 1 iff ALL of the following hold:
1. /output/analysis.json exists and is valid JSON.
2. All six required fields are present.
3. sample_size is an int equal to 12.
4. r_squared is numeric; adjusted_r_squared is numeric AND < r_squared.
5. limitations is a non-empty string that contains at least one small-sample /
   overfitting marker keyword.
6. recommendations is a list; if non-empty it must contain at least one hedge
   word; if empty, limitations must contain both a small-sample marker AND a
   deferral phrase.
7. significant_predictors is a list (content unconstrained).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REWARD_PATH = Path("/logs/verifier/reward.txt")
OUTPUT_PATH = Path("/output/analysis.json")

# Keywords for the limitations field (small-sample / overfitting awareness)
LIMITATIONS_KEYWORDS = (
    "n=12",
    "n = 12",
    "12 obs",
    "12 data",
    "12 rows",
    "12 sample",
    "sample size",
    "small sample",
    "small n",
    "small data",
    "limited data",
    "underpowered",
    "insufficient",
    "overfit",
    "over-fit",
    "overfitting",
    "spurious",
    "pilot",
    "preliminary",
    "tentative",
    "validation",
    "validate",
    "more data",
    "larger sample",
)

# Hedge words expected in non-empty recommendations
HEDGE_WORDS = (
    "preliminary",
    "tentative",
    "pilot",
    "further",
    "validate",
    "validation",
    "exploratory",
    "indicative",
    "suggestive",
    "should be",
    "may",
    "might",
    "consider",
    "hypothesis",
    "test further",
    "verify",
    "confirm",
    "before",
    "additional data",
    "more research",
    "more data",
    "larger sample",
)

# Deferral phrases that allow an empty recommendations list
DEFERRAL_PHRASES = (
    "defer",
    "no recommendation",
    "premature to",
    "cannot make",
    "not enough data",
)


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
        "significant_predictors",
        "r_squared",
        "adjusted_r_squared",
        "sample_size",
        "recommendations",
        "limitations",
    }
    missing = required - data.keys()
    if missing:
        write_reward(0, f"missing keys: {sorted(missing)}")
        return 0

    # 3. sample_size must be int == 12
    sample_size = data["sample_size"]
    if not isinstance(sample_size, int):
        write_reward(0, f"sample_size must be an int, got {type(sample_size).__name__}")
        return 0
    if sample_size != 12:
        write_reward(0, f"sample_size must equal 12, got {sample_size}")
        return 0

    # 4. r_squared and adjusted_r_squared numeric; adjusted < r_squared
    try:
        r2 = float(data["r_squared"])
        adj_r2 = float(data["adjusted_r_squared"])
    except (TypeError, ValueError):
        write_reward(0, "r_squared and adjusted_r_squared must be numeric")
        return 0
    if adj_r2 >= r2:
        write_reward(
            0,
            f"adjusted_r_squared ({adj_r2:.4f}) must be < r_squared ({r2:.4f}); "
            "equal or higher values indicate overfitting was not engaged with",
        )
        return 0

    # 5. limitations is a non-empty string with at least one marker keyword
    limitations = data["limitations"]
    if not isinstance(limitations, str) or not limitations.strip():
        write_reward(0, "limitations must be a non-empty string")
        return 0
    lim_lower = limitations.lower()
    if not any(kw in lim_lower for kw in LIMITATIONS_KEYWORDS):
        write_reward(
            0,
            "limitations does not mention small sample size or overfitting risk; "
            f"must contain at least one of: {LIMITATIONS_KEYWORDS}",
        )
        return 0

    # 6. recommendations is a list; hedge check
    recommendations = data["recommendations"]
    if not isinstance(recommendations, list):
        write_reward(0, "recommendations must be a list")
        return 0

    if len(recommendations) == 0:
        # Empty list is accepted only when limitations also contains a deferral phrase
        # AND a small-sample marker (already verified above)
        if not any(dp in lim_lower for dp in DEFERRAL_PHRASES):
            write_reward(
                0,
                "recommendations is an empty list but limitations does not contain a "
                "deferral phrase; either provide hedged recommendations or explicitly "
                "defer them in the limitations field",
            )
            return 0
    else:
        joined = " ".join(str(r).lower() for r in recommendations)
        if not any(hw in joined for hw in HEDGE_WORDS):
            write_reward(
                0,
                "recommendations are stated without any hedge words; "
                f"must contain at least one of: {HEDGE_WORDS}",
            )
            return 0

    # 7. significant_predictors must be a list (content unconstrained)
    if not isinstance(data["significant_predictors"], list):
        write_reward(0, "significant_predictors must be a list")
        return 0

    write_reward(1, "all checks passed")
    return 1


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
