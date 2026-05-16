"""Generate per-user A/B test data for the underpowered-ab-test task.

Target:
  - 150 users per group.
  - Variant A: exactly 8 conversions (5.3%).
  - Variant B: exactly 12 conversions (8.0%).
  - chi2_contingency p-value in [0.25, 0.45].

Iterates seeds 0..200000 until constraints are met, then writes
./environment/ab_test.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

OUT = Path(__file__).resolve().parent / "environment" / "ab_test.csv"

N_PER_GROUP = 150
TARGET_A = 8   # conversions
TARGET_B = 12  # conversions
TOLERANCE = 1  # allow +/- 1
P_LOW, P_HIGH = 0.25, 0.45


def chi2_p(a_conv: int, b_conv: int) -> float:
    table = np.array(
        [[a_conv, N_PER_GROUP - a_conv], [b_conv, N_PER_GROUP - b_conv]]
    )
    _, p, _, _ = stats.chi2_contingency(table, correction=False)
    return float(p)


def build_df(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rate_a = TARGET_A / N_PER_GROUP
    rate_b = TARGET_B / N_PER_GROUP
    conv_a = rng.binomial(1, rate_a, N_PER_GROUP)
    conv_b = rng.binomial(1, rate_b, N_PER_GROUP)
    user_ids_a = [f"u{i:05d}" for i in range(N_PER_GROUP)]
    user_ids_b = [f"u{i:05d}" for i in range(N_PER_GROUP, 2 * N_PER_GROUP)]
    df_a = pd.DataFrame({"user_id": user_ids_a, "variant": "A", "converted": conv_a})
    df_b = pd.DataFrame({"user_id": user_ids_b, "variant": "B", "converted": conv_b})
    return pd.concat([df_a, df_b], ignore_index=True)


def main() -> None:
    found_seed = None
    found_a_conv = None
    found_b_conv = None
    found_p = None
    found_df = None

    for seed in range(200_001):
        df = build_df(seed)
        a_conv = int(df[df["variant"] == "A"]["converted"].sum())
        b_conv = int(df[df["variant"] == "B"]["converted"].sum())
        if abs(a_conv - TARGET_A) <= TOLERANCE and abs(b_conv - TARGET_B) <= TOLERANCE:
            p = chi2_p(a_conv, b_conv)
            if P_LOW <= p <= P_HIGH:
                found_seed = seed
                found_a_conv = a_conv
                found_b_conv = b_conv
                found_p = p
                found_df = df
                break

    if found_seed is None:
        print("ERROR: No suitable seed found in range 0..200000", file=sys.stderr)
        sys.exit(1)

    # Write CSV
    OUT.parent.mkdir(parents=True, exist_ok=True)
    found_df.to_csv(OUT, index=False)

    # Summary stats
    rate_a = found_a_conv / N_PER_GROUP
    rate_b = found_b_conv / N_PER_GROUP
    effect = rate_b - rate_a

    # Post-hoc power for 3-pp lift at alpha=0.05 with n=150 per group
    lift = 0.03  # 3 percentage-point lift we're trying to detect
    baseline = rate_a
    es = proportion_effectsize(baseline + lift, baseline)
    power_calc = NormalIndPower()
    posthoc_power = power_calc.solve_power(
        effect_size=es, nobs1=N_PER_GROUP, alpha=0.05, ratio=1.0
    )

    # Required n for 80% power at 3-pp lift
    required_n = power_calc.solve_power(
        effect_size=es, power=0.80, alpha=0.05, ratio=1.0
    )

    print(f"Seed found      : {found_seed}")
    print(f"A conversions   : {found_a_conv}/{N_PER_GROUP}  ({rate_a:.1%})")
    print(f"B conversions   : {found_b_conv}/{N_PER_GROUP}  ({rate_b:.1%})")
    print(f"chi2 p-value    : {found_p:.4f}")
    print(f"Effect (B-A)    : {effect:.4f} ({effect:.1%})")
    print(f"Post-hoc power  : {posthoc_power:.4f}  (for 3pp lift, alpha=0.05, n=150)")
    print(f"Required n/group: {int(np.ceil(required_n))}  (for 80% power, 3pp lift)")
    print(f"CSV written to  : {OUT}")


if __name__ == "__main__":
    main()
