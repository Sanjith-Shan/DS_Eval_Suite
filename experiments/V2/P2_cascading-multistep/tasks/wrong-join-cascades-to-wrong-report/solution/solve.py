"""Reference solution for wrong-join-cascades-to-wrong-report.

Correctly uses LEFT join (transactions -> customers) so that orphan
transactions (customer_ids with no matching customer record) are retained
rather than silently dropped.  The subsequent aggregation therefore reflects
the true per-quarter revenue, and Q4 is correctly identified as NOT
underperforming.

Paths are read from environment variables so the script can be tested locally
without Docker:
  WORKSPACE_DIR  (default /workspace)
  OUTPUT_DIR     (default /output)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
OUTPUT    = Path(os.environ.get("OUTPUT_DIR",    "/output"))


def main() -> None:
    # --- load ---
    txns      = pd.read_csv(WORKSPACE / "transactions.csv")
    customers = pd.read_csv(WORKSPACE / "customers.csv")
    regions   = pd.read_csv(WORKSPACE / "regions.csv")

    # --- join: LEFT so orphan transactions are kept ---
    merged = txns.merge(customers, on="customer_id", how="left")
    merged = merged.merge(regions, on="region_id", how="left")

    # --- aggregate quarterly revenue ---
    quarter_order = ["Q1", "Q2", "Q3", "Q4"]
    qrev = (
        merged.groupby("quarter")["amount"]
        .sum()
        .reindex(quarter_order)
        .fillna(0.0)
    )

    total_annual    = float(qrev.sum())
    annual_avg      = total_annual / 4
    threshold       = 0.90 * annual_avg   # 10% below average

    underperforming = [
        q for q in quarter_order
        if float(qrev[q]) < threshold
    ]

    # --- write output ---
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {
        "quarterly_revenue": {q: round(float(qrev[q]), 2) for q in quarter_order},
        "underperforming_quarters": underperforming,
        "total_annual_revenue": round(total_annual, 2),
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote {OUTPUT / 'report.json'}")
    print(f"Quarterly revenue: {report['quarterly_revenue']}")
    print(f"Underperforming:   {report['underperforming_quarters']}")
    print(f"Total annual:      ${report['total_annual_revenue']:,.2f}")


if __name__ == "__main__":
    main()
