"""Reference solution for etl-timezone-schema-merge.

Reads Q1/Q2/Q3, normalises schemas, converts timestamps to UTC (handling the
March DST gap by shifting forward), strips currency from amount, adds the
discount_code column, deduplicates on transaction_id, and writes the clean
CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path("/workspace")
OUT = Path("/output/transactions_clean.csv")


def load_q1() -> pd.DataFrame:
    df = pd.read_csv(WORKSPACE / "Q1.csv")
    df = df.rename(columns={"transaction_date": "raw_timestamp"})
    df["source"] = "Q1"
    df["source_tz"] = "UTC"
    return df


def load_q2() -> pd.DataFrame:
    df = pd.read_csv(WORKSPACE / "Q2.csv")
    df = df.rename(columns={"txn_date": "raw_timestamp"})
    df["source"] = "Q2"
    df["source_tz"] = "US/Eastern"
    return df


def load_q3() -> pd.DataFrame:
    df = pd.read_csv(WORKSPACE / "Q3.csv")
    df = df.rename(columns={"date": "raw_timestamp"})
    df["amount"] = df["amount"].astype(str).str.replace("$", "", regex=False).astype(float)
    df["source"] = "Q3"
    df["source_tz"] = "US/Pacific"
    return df


def to_utc(series: pd.Series, tz: str) -> pd.Series:
    naive = pd.to_datetime(series, errors="raise")
    if tz == "UTC":
        return naive.dt.tz_localize("UTC")
    localised = naive.dt.tz_localize(
        tz,
        ambiguous="NaT",       # safe default for fall-back; we don't expect those here.
        nonexistent="shift_forward",  # required for the March DST gap in Q2.
    )
    return localised.dt.tz_convert("UTC")


def main() -> None:
    q1 = load_q1()
    q2 = load_q2()
    q3 = load_q3()

    frames = []
    for df in (q1, q2, q3):
        df = df.copy()
        df["timestamp_utc"] = to_utc(df["raw_timestamp"], df["source_tz"].iloc[0])
        if "discount_code" not in df.columns:
            df["discount_code"] = np.nan
        else:
            df["discount_code"] = df["discount_code"].replace({"": np.nan})
        df = df[["transaction_id", "customer_id", "amount", "category", "timestamp_utc", "discount_code"]]
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset="transaction_id", keep="first")

    # Format timestamps as ISO-8601 with offset.
    merged["timestamp_utc"] = merged["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
