"""Verifier for confounder-identification.

Awards reward=1 only when the agent (a) correctly rejects the causal claim,
(b) names temperature as the confounder, (c) used some controlled method, and
(d) reports plausible numeric magnitudes consistent with the data.
"""

from __future__ import annotations

import json
import os
import re
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

    required = {"causal_claim", "confounder", "method", "marginal_correlation", "controlled_effect", "recommendation"}
    missing = required - data.keys()
    if missing:
        write_reward(0, f"missing keys: {sorted(missing)}")
        return 0

    # (a) Causal claim must be explicitly false (bool).
    if data["causal_claim"] is not False:
        write_reward(0, f"causal_claim should be false, got {data['causal_claim']!r}")
        return 0

    # (b) Confounder must mention temperature.
    confounder = str(data["confounder"]).lower()
    if "temp" not in confounder:
        write_reward(0, f"confounder did not name temperature: {confounder!r}")
        return 0

    # (c) Method must indicate some form of controlled analysis.
    method = str(data["method"]).lower()
    controlled_terms = (
        "partial",
        "regression",
        "ols",
        "control",
        "stratif",
        "multivariate",
        "adjust",
        "ancova",
        "covariate",
        "linear model",
    )
    if not any(term in method for term in controlled_terms):
        write_reward(0, f"method does not indicate a controlled analysis: {method!r}")
        return 0

    # (d) Numeric sanity: marginal correlation must be strongly positive (>0.5),
    # controlled effect must be much smaller (in absolute value) than marginal.
    try:
        marginal = float(data["marginal_correlation"])
        controlled = float(data["controlled_effect"])
    except (TypeError, ValueError):
        write_reward(0, "marginal_correlation / controlled_effect must be numeric")
        return 0

    if marginal < 0.5:
        write_reward(0, f"marginal_correlation should be ~0.7+, got {marginal:.3f}")
        return 0

    if abs(controlled) >= abs(marginal):
        write_reward(0, f"controlled_effect ({controlled:.3f}) should be much smaller than marginal ({marginal:.3f})")
        return 0

    # (e) Recommendation must not affirmatively endorse banning ice cream.
    # We allow "do not ban" / "don't ban" / etc.; flag only positive endorsements.
    rec = str(data["recommendation"]).lower()
    endorses_ban = re.compile(
        r"(?<!not )(?<!don't )(?<!do not )(?<!shouldn't )(?<!should not )"
        r"\b(ban|prohibit|outlaw)\b"
    )
    if endorses_ban.search(rec) and "do not ban" not in rec and "don't ban" not in rec:
        write_reward(0, f"recommendation appears to endorse banning ice cream: {rec!r}")
        return 0

    write_reward(1, "all checks passed")
    return 1


if __name__ == "__main__":
    sys.exit(0 if main() else 0)  # Always exit 0; reward is signaled via the file.
