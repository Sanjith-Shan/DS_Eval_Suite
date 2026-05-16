"""Verifier for statistical-test-assumptions.

Reward 1 iff: agent checked assumptions and correctly flagged normality and
equal-variance violations; used a non-parametric / Welch alternative (not a
plain one-way ANOVA); reported a non-empty post-hoc test; reported plausible
group medians; and included the unambiguous pairwise differences:
  - D > A
  - D > C
  - A > C
The agent may include additional differences (B-related comparisons are
marginal) but must not claim a wrong-direction inequality between any robust
pair.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


REWARD_PATH = Path("/logs/verifier/reward.txt")
OUTPUT_PATH = Path("/output/analysis.json")

# True medians of the generated data; tolerated absolute error.
TRUE_MEDIANS = {"A": 7.20, "B": 6.14, "C": 6.74, "D": 11.44}
MEDIAN_TOL = 0.5

REQUIRED_PAIRS = [("D", "A"), ("D", "C"), ("A", "C")]
WRONG_DIRECTION_PAIRS = [("A", "D"), ("C", "D"), ("C", "A")]


def write_reward(value: int, reason: str = "") -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(str(value))
    print(f"reward={value} reason={reason}", file=sys.stderr)


PAIR_RE = re.compile(r"\s*([ABCD])\s*([<>])\s*([ABCD])\s*")


def parse_pair(s: str) -> tuple[str, str] | None:
    """Return (greater, lesser) if string parses as 'X>Y' or 'Y<X'."""
    m = PAIR_RE.fullmatch(s)
    if not m:
        return None
    a, op, b = m.group(1), m.group(2), m.group(3)
    if a == b:
        return None
    if op == ">":
        return (a, b)
    return (b, a)


def main() -> int:
    if not OUTPUT_PATH.exists():
        write_reward(0, "missing /output/analysis.json")
        return 0

    try:
        data = json.loads(OUTPUT_PATH.read_text())
    except json.JSONDecodeError as exc:
        write_reward(0, f"invalid JSON: {exc}")
        return 0

    required = {
        "assumptions_checked",
        "normality_violated",
        "equal_variance_violated",
        "test_used",
        "test_p_value",
        "post_hoc_test",
        "group_medians",
        "significant_pairs",
    }
    missing = required - data.keys()
    if missing:
        write_reward(0, f"missing keys: {sorted(missing)}")
        return 0

    if data["assumptions_checked"] is not True:
        write_reward(0, "assumptions_checked must be true")
        return 0
    if data["normality_violated"] is not True:
        write_reward(0, "normality_violated must be true")
        return 0
    if data["equal_variance_violated"] is not True:
        write_reward(0, "equal_variance_violated must be true")
        return 0

    test_used = str(data["test_used"]).lower()
    acceptable_tests = (
        "kruskal",
        "wilcoxon rank",  # related
        "welch",
        "permutation",
        "brown-forsythe",
        "brown forsythe",
        "non-parametric",
        "nonparametric",
        "rank-based",
        "rank based",
        "robust anova",
        "robust one-way",
    )
    if not any(t in test_used for t in acceptable_tests):
        write_reward(0, f"test_used is not an accepted non-parametric / robust alternative: {test_used!r}")
        return 0
    # Reject plain ANOVA without 'welch' qualifier.
    if "anova" in test_used and "welch" not in test_used and "robust" not in test_used:
        write_reward(0, f"plain ANOVA is not appropriate here: {test_used!r}")
        return 0

    try:
        p = float(data["test_p_value"])
    except (TypeError, ValueError):
        write_reward(0, "test_p_value not numeric")
        return 0
    if not 0.0 <= p < 0.05:
        write_reward(0, f"omnibus p-value must be < 0.05 (got {p})")
        return 0

    post_hoc = str(data["post_hoc_test"]).strip()
    if not post_hoc:
        write_reward(0, "post_hoc_test must be non-empty (omnibus is significant)")
        return 0
    # Reject obvious nonsense post-hoc choices that ignore assumption violations.
    if post_hoc.lower() in {"tukey", "tukey hsd", "tukey-hsd"}:
        write_reward(0, "Tukey HSD assumes equal variance and is inappropriate here")
        return 0

    medians = data["group_medians"]
    if not isinstance(medians, dict):
        write_reward(0, "group_medians must be an object")
        return 0
    for store, true_med in TRUE_MEDIANS.items():
        if store not in medians:
            write_reward(0, f"group_medians missing key {store!r}")
            return 0
        try:
            v = float(medians[store])
        except (TypeError, ValueError):
            write_reward(0, f"group_medians[{store!r}] not numeric")
            return 0
        if abs(v - true_med) > MEDIAN_TOL:
            write_reward(0, f"median for {store} off: reported {v}, expected ~{true_med}")
            return 0

    pairs_raw = data["significant_pairs"]
    if not isinstance(pairs_raw, list):
        write_reward(0, "significant_pairs must be a list")
        return 0

    parsed_pairs: set[tuple[str, str]] = set()
    for raw in pairs_raw:
        p2 = parse_pair(str(raw))
        if p2 is None:
            write_reward(0, f"could not parse pair {raw!r}; expected 'X>Y' or 'X<Y'")
            return 0
        parsed_pairs.add(p2)

    for hi, lo in REQUIRED_PAIRS:
        if (hi, lo) not in parsed_pairs:
            write_reward(0, f"missing required pair: {hi}>{lo}")
            return 0
    for wrong_hi, wrong_lo in WRONG_DIRECTION_PAIRS:
        if (wrong_hi, wrong_lo) in parsed_pairs:
            write_reward(0, f"incorrect direction reported: {wrong_hi}>{wrong_lo}")
            return 0

    write_reward(1, "all checks passed")
    return 1


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
