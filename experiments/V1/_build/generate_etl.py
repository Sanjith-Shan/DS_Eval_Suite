"""Generate the three quarterly transaction files plus a README for the
etl-timezone-schema-merge task.

Planted problems:
  1. Column name drift: Q1=transaction_date, Q2=txn_date, Q3=date.
  2. Three different timezones, declared in README only.
  3. Q2 spans the March DST 'spring forward' — some wall-clock timestamps
     fall in the non-existent 02:00-03:00 window and must be handled.
  4. Q3 adds a `discount_code` column that Q1/Q2 lack.
  5. 47 transactions appear in BOTH Q2 and Q3 (same transaction_id).
  6. Q3 amounts are strings prefixed with '$'.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "samples" / "etl-timezone-schema-merge" / "environment"

CATEGORIES = ["groceries", "electronics", "apparel", "home", "books"]
DISCOUNT_CODES = ["SUMMER10", "FLASH20", "VIP15", "WELCOME5"]

SEED = 20260515
N_PER_Q = 3000
N_DUPES = 47


def gen_q1(rng: np.random.Generator) -> pd.DataFrame:
    """Q1: October-December 2023. Times are NAIVE strings already in UTC."""
    start = pd.Timestamp("2023-10-01 00:00:00")
    end = pd.Timestamp("2023-12-31 23:59:59")
    seconds = rng.integers(0, int((end - start).total_seconds()), size=N_PER_Q)
    ts = pd.to_datetime(start) + pd.to_timedelta(seconds, unit="s")
    df = pd.DataFrame(
        {
            "transaction_id": [f"Q1T{100000 + i:07d}" for i in range(N_PER_Q)],
            "customer_id": [f"C{rng.integers(0, 5000):05d}" for _ in range(N_PER_Q)],
            "amount": np.round(rng.uniform(5.0, 250.0, size=N_PER_Q), 2),
            "category": rng.choice(CATEGORIES, size=N_PER_Q),
            "transaction_date": ts.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    return df


def gen_q2(rng: np.random.Generator) -> pd.DataFrame:
    """Q2: January-March 2024 (covers the March 10 DST spring-forward).
    Wall-clock times in US/Eastern, written as naive strings.
    """
    start = pd.Timestamp("2024-01-01 00:00:00")
    end = pd.Timestamp("2024-03-31 23:59:59")
    seconds = rng.integers(0, int((end - start).total_seconds()), size=N_PER_Q - 12)
    ts = pd.to_datetime(start) + pd.to_timedelta(seconds, unit="s")

    # Plant 12 rows inside the non-existent 02:00-03:00 window on 2024-03-10.
    dst_dt = pd.to_datetime("2024-03-10 02:00:00")
    dst_offsets_min = rng.integers(0, 60, size=12)
    dst_ts = dst_dt + pd.to_timedelta(dst_offsets_min, unit="m")
    ts = pd.concat([pd.Series(ts), pd.Series(dst_ts)]).reset_index(drop=True)
    order = np.argsort(ts.values)
    ts = ts.iloc[order].reset_index(drop=True)

    df = pd.DataFrame(
        {
            "transaction_id": [f"Q2T{200000 + i:07d}" for i in range(N_PER_Q)],
            "customer_id": [f"C{rng.integers(0, 5000):05d}" for _ in range(N_PER_Q)],
            "amount": np.round(rng.uniform(5.0, 250.0, size=N_PER_Q), 2),
            "category": rng.choice(CATEGORIES, size=N_PER_Q),
            "txn_date": ts.dt.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    return df


def gen_q3(rng: np.random.Generator, q2_dupe_ids: list[str]) -> pd.DataFrame:
    """Q3: April-June 2024. US/Pacific wall time. `amount` as dollar-prefixed
    strings. Carries a new `discount_code` column. 47 transaction_ids overlap
    with Q2.
    """
    start = pd.Timestamp("2024-04-01 00:00:00")
    end = pd.Timestamp("2024-06-30 23:59:59")
    seconds = rng.integers(0, int((end - start).total_seconds()), size=N_PER_Q)
    ts = pd.to_datetime(start) + pd.to_timedelta(seconds, unit="s")

    tx_ids = [f"Q3T{300000 + i:07d}" for i in range(N_PER_Q)]
    # Overwrite the first 47 with the chosen dupe ids.
    for i, dup in enumerate(q2_dupe_ids):
        tx_ids[i] = dup

    amounts = rng.uniform(5.0, 250.0, size=N_PER_Q)
    amount_strs = [f"${a:.2f}" for a in amounts]

    df = pd.DataFrame(
        {
            "transaction_id": tx_ids,
            "customer_id": [f"C{rng.integers(0, 5000):05d}" for _ in range(N_PER_Q)],
            "amount": amount_strs,
            "category": rng.choice(CATEGORIES, size=N_PER_Q),
            "date": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "discount_code": rng.choice(DISCOUNT_CODES + [""], size=N_PER_Q),
        }
    )
    return df


README = """# Transaction Data Notes

This directory contains three quarterly transaction extracts produced by
different downstream teams. They predate the data-platform unification, so
schemas drift across files. Please consolidate them into a single clean
dataset.

## Timezones (very important)

- `Q1.csv` timestamps are recorded in **UTC**.
- `Q2.csv` timestamps are recorded in **US/Eastern** wall time. Daylight Saving
  Time begins on 2024-03-10; the 02:00-03:00 window on that day does not exist
  in local time.
- `Q3.csv` timestamps are recorded in **US/Pacific** wall time.

## Schema notes

- Q1 calls the timestamp column `transaction_date`, Q2 calls it `txn_date`,
  and Q3 calls it `date`. They all mean the same thing.
- Q3 introduces a `discount_code` column not present in Q1 or Q2. Use NaN /
  null for rows from the earlier quarters.
- Q3 stores `amount` as a dollar-prefixed string (`"$45.99"`). Q1 and Q2 store
  it as a float.

## Duplicate handling

A backfill job ran against both Q2 and Q3, so 47 transactions appear in both
files (identical `transaction_id`). Deduplicate on `transaction_id`, keeping
the first occurrence.
"""


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    q1 = gen_q1(rng)
    q2 = gen_q2(rng)

    # Pick the 47 dupe ids from Q2.
    dupe_ids = list(rng.choice(q2["transaction_id"].values, size=N_DUPES, replace=False))
    q3 = gen_q3(rng, dupe_ids)

    q1.to_csv(OUT_DIR / "Q1.csv", index=False)
    q2.to_csv(OUT_DIR / "Q2.csv", index=False)
    q3.to_csv(OUT_DIR / "Q3.csv", index=False)
    (OUT_DIR / "README.md").write_text(README)

    expected_total = 3 * N_PER_Q - N_DUPES
    print(f"Q1={len(q1)} Q2={len(q2)} Q3={len(q3)} expected_after_dedupe={expected_total}")


if __name__ == "__main__":
    main()
