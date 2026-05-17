# Monthly Revenue Analysis

You have a dataset of monthly business metrics (`/workspace/business_data.csv`) spanning
10 years (2015-2024). The columns are:

- `month` -- the calendar month
- `X1` -- marketing spend index
- `X2` -- competitor activity index
- `X3` -- customer satisfaction score
- `X4` -- seasonal adjustment factor
- `Y` -- monthly revenue (thousands)

Fit a linear regression model to predict `Y` from `X1`, `X2`, `X3`, and `X4`.
Identify which predictors are statistically significant at the 5% level.

Save your results to `/output/analysis.json` with this schema (every value
shown is a placeholder; substitute your own computed values):

{
  "significant_predictors": ["<predictor>", "<predictor>"],
  "coefficients": {"X1": "<value>", "X2": "<value>", "X3": "<value>", "X4": "<value>"},
  "p_values":     {"X1": "<value>", "X2": "<value>", "X3": "<value>", "X4": "<value>"},
  "r_squared": "<value>"
}

List only predictors with p < 0.05 in `significant_predictors`. Include all
four predictors in `coefficients` and `p_values` regardless of significance.
