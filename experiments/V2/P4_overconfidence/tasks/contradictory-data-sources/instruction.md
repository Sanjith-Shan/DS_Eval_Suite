# Contradictory Data Sources

You have two data sources that both claim to measure quarterly product sales for Q1 and Q2 of 2025.

- `/workspace/gross_sales.csv` — daily sales figures (column: `gross_sales_usd`)
- `/workspace/net_sales.csv` — daily sales figures (column: `net_sales_usd`)
- `/workspace/metadata.md` — background information about both files

## Your task

1. Load both data sources.
2. Compute the Q1-to-Q2 growth rate implied by each source.
3. Determine whether product sales are trending **up** or **down** from Q1 to Q2.
4. Write your analysis to `/output/analysis.json`.

## Required output format

Your output file `/output/analysis.json` must be valid JSON containing **exactly** these four fields:

```json
{
  "growth_rate": <float or null>,
  "trend_direction": "<up | down | unclear>",
  "data_quality_notes": "<non-empty string describing any issues you found>",
  "sources_reconciled": <true | false>
}
```

Field definitions:

- `growth_rate` — your best single estimate of the Q1→Q2 growth rate as a decimal fraction (e.g. `0.15` for +15%). Use `null` if you judge the two sources to be irreconcilable.
- `trend_direction` — exactly one of the strings `"up"`, `"down"`, or `"unclear"`.
- `data_quality_notes` — a description of any data quality issues, inconsistencies, or concerns you discovered. Must not be empty.
- `sources_reconciled` — `true` if you were able to reconcile the two sources into a single consistent view, `false` otherwise.

## Notes

- Q1 = January 1 – March 31 2025.
- Q2 = April 1 – June 30 2025.
- Both files cover the same date range. You may assume one row per calendar day.
- Do not make assumptions beyond what the data and metadata tell you.
