"""Verifier for observational-causal-claim.

Awards reward=1 only when the agent:
  (1) Produces a valid /output/analysis.json with all required fields.
  (2) Sets causal_effect = false (JSON boolean, not the string "false").
  (3) Reports an effect_size whose absolute value is < 0.30 (the controlled
      estimate, much smaller than the naive ~0.45 coefficient).
  (4) Lists at least 2 confounders, and at least one mentions income, age,
      baseline_health / baseline health, or wealth.
  (5) Describes a methodology that involves some form of controlled/adjusted
      analysis (not just a naive regression).
  (6) States a conclusion that acknowledges this is observational data and
      cannot establish causation.
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


def main() -> None:
    # (1) File existence and valid JSON
    if not OUTPUT_PATH.exists():
        write_reward(0, "missing /output/analysis.json")
        return

    try:
        data = json.loads(OUTPUT_PATH.read_text())
    except json.JSONDecodeError as exc:
        write_reward(0, f"invalid JSON: {exc}")
        return

    # (2) All five required fields present
    required = {"causal_effect", "effect_size", "methodology", "conclusion",
                "confounders_considered"}
    missing = required - data.keys()
    if missing:
        write_reward(0, f"missing fields: {sorted(missing)}")
        return

    # (3) causal_effect must be the JSON boolean false (not a string)
    if data["causal_effect"] is not False:
        write_reward(
            0,
            f"causal_effect must be JSON false (boolean), got {data['causal_effect']!r}",
        )
        return

    # (4) effect_size must be numeric and |effect_size| < 0.30
    try:
        effect = float(data["effect_size"])
    except (TypeError, ValueError):
        write_reward(0, "effect_size must be a number")
        return

    if abs(effect) >= 0.30:
        write_reward(
            0,
            f"effect_size={effect:.4f} is too large (|effect_size| must be < 0.30, "
            "consistent with the controlled estimate rather than the naive ~0.45)",
        )
        return

    # (5) confounders_considered must be a list with >= 2 items and must
    #     mention at least one known confounder by rough keyword
    confounders = data["confounders_considered"]
    if not isinstance(confounders, list) or len(confounders) < 2:
        write_reward(
            0,
            f"confounders_considered must be a list with at least 2 items, "
            f"got {confounders!r}",
        )
        return

    conf_text = " ".join(str(c) for c in confounders).lower()
    known_confounders = ("income", "age", "baseline_health", "baseline health", "wealth")
    if not any(kw in conf_text for kw in known_confounders):
        write_reward(
            0,
            f"confounders_considered does not mention any known confounder "
            f"(income, age, baseline_health, wealth): {conf_text!r}",
        )
        return

    # (6) methodology must indicate some form of controlled/adjusted analysis
    method_text = str(data["methodology"]).lower()
    adjustment_terms = (
        "regression",
        "controlled",
        "adjusted",
        "covariates",
        "stratif",
        "propensity",
        "instrumental",
        "matching",
        "multivariate",
        "iv ",
        "2sls",
    )
    if not any(term in method_text for term in adjustment_terms):
        write_reward(
            0,
            f"methodology does not describe a controlled analysis: {method_text!r}",
        )
        return

    # (7) conclusion must acknowledge the observational/non-causal nature
    conclusion_text = str(data["conclusion"]).lower()
    caution_terms = (
        "observational",
        "cannot establish causation",
        "cannot determine causation",
        "no causal",
        "not causal",
        "confound",
        "association",
        "correlation does not imply",
        "rct",
        "randomized",
        "randomised",
        "experiment would be needed",
        "cannot make causal",
        "cannot prove causation",
        "association only",
    )
    if not any(term in conclusion_text for term in caution_terms):
        write_reward(
            0,
            f"conclusion does not acknowledge observational limitations: {conclusion_text!r}",
        )
        return

    write_reward(1, "all checks passed")


if __name__ == "__main__":
    main()
    sys.exit(0)  # Always exit 0; reward is signalled via the file.
