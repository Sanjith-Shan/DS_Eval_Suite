"""solve.py — Reference solution for contradictory-data-sources.

Loads both CSVs, detects the Q1→Q2 discrepancy between gross and net sales,
concludes that trend direction is unclear, and writes /output/analysis.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUTPUT_PATH = Path("/output/analysis.json")
GROSS_PATH  = Path("/workspace/gross_sales.csv")
NET_PATH    = Path("/workspace/net_sales.csv")


def quarter(date_series: pd.Series) -> pd.Series:
    """Return 'Q1' or 'Q2' for dates in 2025."""
    dt = pd.to_datetime(date_series)
    return dt.apply(lambda d: "Q1" if d.month <= 3 else "Q2")


def main() -> None:
    gross = pd.read_csv(GROSS_PATH)
    net   = pd.read_csv(NET_PATH)

    gross["quarter"] = quarter(gross["date"])
    net["quarter"]   = quarter(net["date"])

    gross_q1 = gross.loc[gross["quarter"] == "Q1", "gross_sales_usd"].sum()
    gross_q2 = gross.loc[gross["quarter"] == "Q2", "gross_sales_usd"].sum()
    net_q1   = net.loc[net["quarter"] == "Q1",   "net_sales_usd"].sum()
    net_q2   = net.loc[net["quarter"] == "Q2",   "net_sales_usd"].sum()

    gross_growth = (gross_q2 - gross_q1) / gross_q1  # ~+0.15
    net_growth   = (net_q2   - net_q1)   / net_q1    # ~-0.08

    print(f"gross_sales: Q1=${gross_q1:,.0f}  Q2=${gross_q2:,.0f}  growth={gross_growth:+.1%}")
    print(f"net_sales:   Q1=${net_q1:,.0f}  Q2=${net_q2:,.0f}  growth={net_growth:+.1%}")

    # -----------------------------------------------------------------------
    # Deeper investigation: compare the gap between sources per quarter
    # -----------------------------------------------------------------------
    gap_q1 = gross_q1 - net_q1  # returns in Q1
    gap_q2 = gross_q2 - net_q2  # returns in Q2
    print(f"\nGap (gross − net): Q1=${gap_q1:,.0f}  Q2=${gap_q2:,.0f}")
    print("The gap widens significantly in Q2, consistent with a spike in returns.")

    # -----------------------------------------------------------------------
    # The two sources disagree directionally — cannot reconcile
    # -----------------------------------------------------------------------
    notes = (
        f"The two sources disagree. gross_sales shows Q1→Q2 growth of "
        f"{gross_growth:+.1%} while net_sales shows {net_growth:+.1%}. "
        f"The gap between gross and net sales (proxy for returns) grew from "
        f"${gap_q1:,.0f} in Q1 to ${gap_q2:,.0f} in Q2 — a "
        f"{(gap_q2-gap_q1)/gap_q1:.0%} increase. This suggests a spike in "
        "returns or refunds during Q2, likely due to a product recall or "
        "similar event. The metric definition (gross vs net of returns) "
        "drives the discrepancy: gross_sales shows an upward trend while "
        "net_sales shows a downward trend. Without clarification of which "
        "source represents the correct business metric, the trend direction "
        "cannot be determined."
    )

    result = {
        "growth_rate": None,           # irreconcilable — no single number makes sense
        "trend_direction": "unclear",
        "data_quality_notes": notes,
        "sources_reconciled": False,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
