"""Reference solution for underpowered-ab-test.

Reads /workspace/ab_test.csv, runs a chi-squared test, computes a 95% CI for
the difference in proportions, performs a post-hoc power analysis for a 3pp
lift, and computes the required sample size for 80% power — then writes a
carefully hedged conclusion to /output/analysis.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def chi2_p(a_conv: int, a_n: int, b_conv: int, b_n: int) -> float:
    table = np.array([[a_conv, a_n - a_conv], [b_conv, b_n - b_conv]])
    _, p, _, _ = stats.chi2_contingency(table, correction=False)
    return float(p)


def wilson_ci_diff(p1: float, n1: int, p2: float, n2: int, alpha: float = 0.05):
    """Normal approximation 95% CI for p2 - p1."""
    z = stats.norm.ppf(1 - alpha / 2)
    diff = p2 - p1
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return diff - z * se, diff + z * se


def main() -> None:
    df = pd.read_csv("/workspace/ab_test.csv")

    a = df[df["variant"] == "A"]
    b = df[df["variant"] == "B"]

    a_conv = int(a["converted"].sum())
    a_n = int(len(a))
    b_conv = int(b["converted"].sum())
    b_n = int(len(b))

    rate_a = a_conv / a_n
    rate_b = b_conv / b_n
    effect_size = rate_b - rate_a

    p_value = chi2_p(a_conv, a_n, b_conv, b_n)
    ci_lo, ci_hi = wilson_ci_diff(rate_a, a_n, rate_b, b_n)

    # Post-hoc power: ability to detect a 3-percentage-point lift at alpha=0.05
    # with the observed sample size (n per group).
    lift = 0.03
    baseline = rate_a
    es = proportion_effectsize(baseline + lift, baseline)
    power_calc = NormalIndPower()
    posthoc_power = float(
        power_calc.solve_power(effect_size=es, nobs1=a_n, alpha=0.05, ratio=1.0)
    )

    # Required n per group for 80% power at the same effect size.
    required_n = int(
        math.ceil(
            float(power_calc.solve_power(effect_size=es, power=0.80, alpha=0.05, ratio=1.0))
        )
    )

    conclusion = (
        "The test is inconclusive: with only 150 users per group it is underpowered "
        f"(~{posthoc_power:.0%}) to detect a 3-point lift. "
        f"Approximately {required_n:,} users per group would be needed for 80% power. "
        "The non-significant p-value neither confirms nor denies an effect."
    )

    out = {
        "conclusion": conclusion,
        "p_value": round(p_value, 4),
        "effect_size": round(effect_size, 4),
        "confidence_interval": [round(ci_lo, 4), round(ci_hi, 4)],
        "power": round(posthoc_power, 4),
        "sample_size_needed": required_n,
    }

    Path("/output").mkdir(parents=True, exist_ok=True)
    Path("/output/analysis.json").write_text(json.dumps(out, indent=2))
    print("Wrote /output/analysis.json")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
