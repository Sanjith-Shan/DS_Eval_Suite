# Daily Sales Forecast (Regime Change)

`/workspace/sales.csv` contains three years of daily store sales (columns
`date` and `sales`). The store underwent a renovation roughly midway through
year two that changed the baseline level. Your job is to forecast daily sales
for the next 30 days starting the day after the last observation.

Write the forecast to `/output/forecast.csv` with columns:

```
date,predicted_sales
```

Requirements:

- Exactly 30 rows, one per consecutive day immediately following the last
  observation in `sales.csv`.
- `date` formatted as `YYYY-MM-DD`.
- `predicted_sales` must be numeric and non-negative.
- The mean absolute percentage error (MAPE) against the held-out actuals will
  be evaluated; choose a model that captures the post-renovation level and
  the weekly / yearly seasonality.

Make sure `/output` exists before writing.
