# Quarterly Transaction Consolidation

Three CSV files under `/workspace/` (`Q1.csv`, `Q2.csv`, `Q3.csv`) need to be
merged into a single clean dataset at `/output/transactions_clean.csv`. A
`/workspace/README.md` describes the schema drift and timezone conventions —
read it before you write code.

The output CSV must have exactly these columns in this order:

```
transaction_id, customer_id, amount, category, timestamp_utc, discount_code
```

Requirements:

- `timestamp_utc` must be an ISO-8601 timestamp in UTC (e.g. `2024-03-10T07:30:00+00:00`).
- `amount` must be a numeric float (no currency symbols, no quoting).
- `discount_code` must be present with a missing value (empty string or NaN)
  for rows that came from Q1 / Q2.
- Each `transaction_id` must appear exactly once (deduplicate, keeping the
  first occurrence in input order Q1 → Q2 → Q3).
- Timestamps that fall in the non-existent DST window must be present in the
  output (do not drop them); shifting them to the next valid UTC instant is
  acceptable.

Make sure `/output` exists before writing.
