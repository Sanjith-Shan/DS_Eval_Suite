# Transaction Data Notes

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
