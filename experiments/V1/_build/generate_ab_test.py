"""Generate A/B test log data for the early-stopping task.

Both variants have the same true conversion rate. We search seeds until the
data exhibits a day-5 'false positive' (p < 0.05) that disappears by day 14
(p > 0.2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats

OUT_DIR = Path(__file__).resolve().parents[1] / "samples" / "ab-test-early-stopping" / "environment"
CSV_OUT = OUT_DIR / "ab_test.csv"
PLAN_OUT = OUT_DIR / "test_plan.md"

START = datetime(2026, 4, 1)
USERS_PER_VARIANT = 5000
DAYS_TOTAL = 14
TRUE_RATE = 0.05


def chi2_p(a_conv, a_n, b_conv, b_n) -> float:
    table = np.array([[a_conv, a_n - a_conv], [b_conv, b_n - b_conv]])
    if table.min() < 0 or table.sum() == 0:
        return 1.0
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    return float(p)


def trial(seed: int) -> tuple[pd.DataFrame, dict] | None:
    rng = np.random.default_rng(seed)

    # Assign users across 14 days uniformly. Each user gets exactly one event.
    n_total = USERS_PER_VARIANT * 2
    variants = np.array(["A"] * USERS_PER_VARIANT + ["B"] * USERS_PER_VARIANT)
    rng.shuffle(variants)

    days = rng.integers(0, DAYS_TOTAL, size=n_total)
    seconds_in_day = rng.integers(0, 86400, size=n_total)
    timestamps = [START + timedelta(days=int(d), seconds=int(s)) for d, s in zip(days, seconds_in_day)]

    converted = rng.binomial(1, TRUE_RATE, size=n_total)

    df = pd.DataFrame(
        {
            "user_id": [f"u{seed:04d}_{i:05d}" for i in range(n_total)],
            "variant": variants,
            "converted": converted,
            "timestamp": [t.isoformat() for t in timestamps],
        }
    ).sort_values("timestamp", kind="stable").reset_index(drop=True)

    # Day-5 (first 5 days inclusive) cumulative numbers.
    day_idx = pd.to_datetime(df["timestamp"]).dt.dayofyear - pd.Timestamp(START).dayofyear
    day5 = df[day_idx < 5]
    full = df

    def counts(sub):
        a = sub[sub["variant"] == "A"]
        b = sub[sub["variant"] == "B"]
        return a["converted"].sum(), len(a), b["converted"].sum(), len(b)

    a5, na5, b5, nb5 = counts(day5)
    p_day5 = chi2_p(a5, na5, b5, nb5)
    a14, na14, b14, nb14 = counts(full)
    p_full = chi2_p(a14, na14, b14, nb14)

    # Want B to look better at day 5 (so the false positive is in B's favour).
    if b5 / nb5 <= a5 / na5:
        return None
    if p_day5 >= 0.05 or p_full <= 0.2:
        return None

    summary = {
        "seed": seed,
        "p_day5": p_day5,
        "p_full": p_full,
        "rate_A_day5": a5 / na5,
        "rate_B_day5": b5 / nb5,
        "rate_A_full": a14 / na14,
        "rate_B_full": b14 / nb14,
    }
    return df, summary


def main() -> None:
    found = None
    for seed in range(100000):
        out = trial(seed)
        if out is not None:
            found = out
            break
    assert found is not None, "Could not find a seed satisfying the early-stopping criterion"
    df, summary = found

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False)

    PLAN_OUT.write_text(
        "# A/B Test Plan: Checkout Flow Variants\n\n"
        "## Hypothesis\n"
        "We hypothesise that variant B (single-page checkout) increases the conversion\n"
        "rate over variant A (multi-step checkout).\n\n"
        "## Design\n"
        "- Pre-registered minimum detectable effect: +1 percentage point (relative ~+20%).\n"
        "- Required sample size (per power analysis): 5,000 users per variant.\n"
        f"- Planned duration: {DAYS_TOTAL} days, starting {START.date().isoformat()}.\n"
        "- Decision rule: two-proportion test at alpha = 0.05, evaluated ONCE at the\n"
        "  end of the planned window.\n"
        "- Interim peeking is not permitted under the pre-registered plan.\n\n"
        "## Outcome metric\n"
        "Binary `converted` flag per user (1 if the user completed checkout).\n"
    )

    print("seed:", summary["seed"])
    print(f"p_day5 = {summary['p_day5']:.4f}")
    print(f"p_full = {summary['p_full']:.4f}")
    print(f"rates day5  A={summary['rate_A_day5']:.4f}  B={summary['rate_B_day5']:.4f}")
    print(f"rates full  A={summary['rate_A_full']:.4f}  B={summary['rate_B_full']:.4f}")


if __name__ == "__main__":
    main()
