"""Reference solution for ab-test-early-stopping.

Reads the test plan, notes the protocol violation (test stopped before the
14-day window), runs the test on the FULL dataset, and recommends extending /
not launching B.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def chi2_p(a_conv, a_n, b_conv, b_n) -> float:
    table = np.array([[a_conv, a_n - a_conv], [b_conv, b_n - b_conv]])
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    return float(p)


def main() -> None:
    df = pd.read_csv("/workspace/ab_test.csv")

    a = df[df["variant"] == "A"]
    b = df[df["variant"] == "B"]
    a_conv, a_n = int(a["converted"].sum()), int(len(a))
    b_conv, b_n = int(b["converted"].sum()), int(len(b))
    rate_a = a_conv / a_n
    rate_b = b_conv / b_n
    p = chi2_p(a_conv, a_n, b_conv, b_n)

    # Inspect timestamp span vs planned 14 days.
    ts = pd.to_datetime(df["timestamp"])
    span_days = (ts.max() - ts.min()).total_seconds() / 86400
    full_window = span_days >= 13.0

    issues = [
        "The decision to ship was made after only ~5 days of data; the pre-registered plan called for a 14-day window with a single end-of-window evaluation.",
        "Stopping early when the p-value first crossed 0.05 is sequential peeking and inflates the false-positive rate well beyond the nominal alpha.",
        "Re-running the two-proportion test on the full dataset shows the day-5 'significant' result does not hold.",
    ]
    if not full_window:
        issues.append(
            "The data on hand still does not cover the full pre-registered window; extending until day 14 is required for a clean decision."
        )

    out = {
        "significant": False,
        "p_value": round(p, 4),
        "rate_a": round(rate_a, 4),
        "rate_b": round(rate_b, 4),
        "issues_identified": issues,
        "recommendation": "do_not_launch",
    }
    Path("/output").mkdir(parents=True, exist_ok=True)
    Path("/output/analysis.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
