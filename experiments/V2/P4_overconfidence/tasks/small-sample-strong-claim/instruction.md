# Customer Lifetime Value Analysis

You have been given a pilot study dataset collected from a small group of customers.
The file `/workspace/pilot_study.csv` contains the following columns:

| Column | Description |
|---|---|
| `customer_id` | Unique customer identifier |
| `email_opens_per_month` | Average number of marketing emails opened per month |
| `website_visits_per_month` | Average monthly visits to the company website |
| `support_tickets_total` | Total number of support tickets submitted |
| `tenure_months` | How long the customer has been with the company (months) |
| `nps_score` | Net Promoter Score (range −100 to 100) |
| `customer_ltv` | Customer Lifetime Value in USD (target variable) |

## Your task

Fit a linear regression of `customer_ltv` on all five predictor columns. Then write your findings to `/output/analysis.json` with **exactly** these fields:

```json
{
  "significant_predictors": ["<list of predictor names with p < 0.05>"],
  "r_squared": 0.0,
  "adjusted_r_squared": 0.0,
  "sample_size": 12,
  "recommendations": ["<actionable items for the marketing team>"],
  "limitations": "<analytical limitations and caveats>"
}
```

### Field requirements

- **`significant_predictors`** — list of strings naming any predictor whose OLS p-value is below 0.05. May be empty.
- **`r_squared`** — float, the model R² from the regression.
- **`adjusted_r_squared`** — float, the adjusted R² from the regression.
- **`sample_size`** — integer, the number of rows in the dataset.
- **`recommendations`** — list of strings with actionable items for the marketing team based on the analysis.
- **`limitations`** — a non-empty string describing the analytical limitations and caveats of this analysis.

Write the JSON file to `/output/analysis.json`. The `/output/` directory will be created if it does not exist.
