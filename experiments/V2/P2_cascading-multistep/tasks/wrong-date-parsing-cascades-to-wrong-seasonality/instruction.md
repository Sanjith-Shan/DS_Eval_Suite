# Sales Seasonal Analysis

You are given 3 years of daily sales data (2022-01-01 through 2024-12-31) in `/workspace/sales.csv`. The file has three columns: `date` (daily dates as strings), `sales` (daily revenue as a float), and `source` (origin of the record).

Your goal is to understand the seasonal pattern in sales and produce a short-term forecast:

1. Parse the `date` column and aggregate sales to monthly totals.
2. Fit an additive seasonal decomposition (period = 12 months) to the monthly series.
3. From the seasonal component, identify the **peak month** (highest average seasonal value, integer 1–12) and the **trough month** (lowest average seasonal value, integer 1–12).
4. Compute the **seasonal strength** as the ratio of the seasonal component's standard deviation to the observed series' standard deviation.
5. Produce a **3-month forecast** for the three months immediately following the data window (January, February, March 2025). A reasonable approach is to use the last trend value plus the average seasonal effect for each target month.

Write your results to `/output/forecast.json` (create `/output/` if it does not exist) with exactly the following structure:

```json
{
  "peak_month": <int, 1-12>,
  "trough_month": <int, 1-12>,
  "forecast_next_3_months": [<float>, <float>, <float>],
  "seasonal_strength": <float>
}
```

`forecast_next_3_months` must be a list of exactly 3 floats, in order: January 2025, February 2025, March 2025.
