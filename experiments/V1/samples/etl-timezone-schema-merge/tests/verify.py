"""Verifier for etl-timezone-schema-merge.

The verifier loads /workspace/Q*.csv to recompute ground-truth row counts and
the DST-gap transaction-id list, then compares against the agent's
/output/transactions_clean.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REWARD_PATH = Path("/logs/verifier/reward.txt")
OUTPUT_PATH = Path("/output/transactions_clean.csv")

EXPECTED_COLUMNS = [
    "transaction_id",
    "customer_id",
    "amount",
    "category",
    "timestamp_utc",
    "discount_code",
]


def write_reward(value: int, reason: str = "") -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(str(value))
    print(f"reward={value} reason={reason}", file=sys.stderr)


def main() -> int:
    if not OUTPUT_PATH.exists():
        write_reward(0, "missing /output/transactions_clean.csv")
        return 0

    try:
        out = pd.read_csv(OUTPUT_PATH, dtype={"discount_code": str})
    except Exception as exc:
        write_reward(0, f"could not read output CSV: {exc}")
        return 0

    if list(out.columns) != EXPECTED_COLUMNS:
        write_reward(0, f"columns wrong; got {list(out.columns)}, expected {EXPECTED_COLUMNS}")
        return 0

    # Load source data for ground truth.
    q1 = pd.read_csv("/workspace/Q1.csv")
    q2 = pd.read_csv("/workspace/Q2.csv")
    q3 = pd.read_csv("/workspace/Q3.csv")

    q2_ids = set(q2["transaction_id"])
    q3_ids = set(q3["transaction_id"])
    dupes = q2_ids & q3_ids
    expected_rows = len(q1) + len(q2) + len(q3) - len(dupes)

    if len(out) != expected_rows:
        write_reward(0, f"row count mismatch: got {len(out)}, expected {expected_rows} ({len(dupes)} dupes)")
        return 0

    # transaction_id uniqueness.
    if out["transaction_id"].duplicated().any():
        n_dup = int(out["transaction_id"].duplicated().sum())
        write_reward(0, f"output has {n_dup} duplicate transaction_id values")
        return 0

    # Amount numeric.
    try:
        amounts = pd.to_numeric(out["amount"], errors="raise")
    except Exception as exc:
        write_reward(0, f"amount column not numeric: {exc}")
        return 0
    if (amounts <= 0).any() or amounts.isna().any():
        write_reward(0, "amount column contains non-positive or NaN values")
        return 0

    # Timestamp parsing — must be timezone-aware UTC.
    try:
        ts = pd.to_datetime(out["timestamp_utc"], utc=False)
    except Exception as exc:
        write_reward(0, f"timestamp_utc not parseable: {exc}")
        return 0
    if ts.dt.tz is None:
        write_reward(0, "timestamp_utc must be timezone-aware (use a +00:00 / Z offset)")
        return 0
    # tz-aware: convert to UTC and ensure offsets all zero.
    try:
        ts_utc = ts.dt.tz_convert("UTC")
    except Exception as exc:
        write_reward(0, f"could not convert to UTC: {exc}")
        return 0
    offsets = ts.apply(lambda x: x.utcoffset().total_seconds() if x is not pd.NaT else 0)
    if (offsets != 0).any():
        n_off = int((offsets != 0).sum())
        write_reward(0, f"{n_off} timestamps are not in UTC offset 0")
        return 0

    # discount_code must be present for ALL rows in EXACTLY the Q3 source.
    q3_only_ids = q3_ids - q2_ids
    out_by_id = out.set_index("transaction_id")

    # Rows originating from Q1 should have null/empty discount_code.
    q1_present = out_by_id.loc[list(q1["transaction_id"])].copy()
    if q1_present["discount_code"].notna().any():
        n = int(q1_present["discount_code"].notna().sum())
        # tolerate empty-string sentinel
        bad = q1_present.loc[q1_present["discount_code"].notna() & (q1_present["discount_code"].astype(str) != "")]
        if len(bad) > 0:
            write_reward(0, f"{len(bad)} Q1-origin rows have non-null discount_code")
            return 0

    # DST-gap transaction-ids must all survive. Recover them by re-parsing Q2.
    q2_dst = q2[q2["txn_date"].str.startswith("2024-03-10 02")]
    dst_ids = set(q2_dst["transaction_id"])
    out_ids = set(out["transaction_id"])
    missing_dst = dst_ids - out_ids
    if missing_dst:
        write_reward(0, f"{len(missing_dst)} DST-gap rows were dropped from output")
        return 0

    # Ensure the timestamps of Q2 DST-gap rows landed in the post-DST UTC band.
    # For 2024-03-10 02:XX US/Eastern shifted forward, valid UTC is 07:00 onward.
    dst_out = out_by_id.loc[list(dst_ids)]
    parsed = pd.to_datetime(dst_out["timestamp_utc"], utc=True)
    bad_dst = parsed[(parsed < "2024-03-10T06:00:00+00:00") | (parsed > "2024-03-10T08:00:00+00:00")]
    if len(bad_dst) > 0:
        write_reward(0, f"{len(bad_dst)} DST-gap rows have implausible UTC timestamps (out of 06-08Z window)")
        return 0

    write_reward(1, f"all checks passed; {len(out)} rows")
    return 1


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
