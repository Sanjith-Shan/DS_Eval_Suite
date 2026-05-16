# Quarterly Revenue Analysis

You are a data analyst tasked with producing a quarterly revenue report for 2024. Three CSV files are available in `/workspace/`:

- `transactions.csv` — every recorded transaction for the year, with columns: `transaction_id`, `customer_id`, `amount`, `date`, `quarter`.
- `customers.csv` — the customer master table, with columns: `customer_id`, `region_id`, `signup_date`.
- `regions.csv` — region reference data, with columns: `region_id`, `region_name`, `country`.

Combine the three tables into a single analytical dataset, compute total revenue for each calendar quarter (Q1–Q4) summing all transaction amounts, and identify any quarter whose revenue falls more than 10% below the annual average. The annual average is defined as `total_annual_revenue / 4`.

Write your findings to `/output/report.json` with exactly the following structure:

```json
{
  "quarterly_revenue": {
    "Q1": <float>,
    "Q2": <float>,
    "Q3": <float>,
    "Q4": <float>
  },
  "underperforming_quarters": ["<quarter>", ...],
  "total_annual_revenue": <float>
}
```

Where:
- `quarterly_revenue` is a dictionary mapping each quarter label to its total revenue as a float.
- `underperforming_quarters` is a list of quarter labels (e.g. `["Q2"]`) for any quarter whose revenue is more than 10% below the annual average; use an empty list `[]` if none qualify.
- `total_annual_revenue` is the sum of all four quarterly revenues as a float.

Make sure the `/output/` directory exists before writing. All numeric values should be plain floats (not strings).
