# Monthly Revenue Analysis

You have a dataset of monthly business metrics (`business_data.csv`) spanning
10 years (2015-2024). The columns are:

- `month` -- the calendar month
- `X1` -- marketing spend index
- `X2` -- competitor activity index
- `X3` -- customer satisfaction score
- `X4` -- seasonal adjustment factor
- `Y` -- monthly revenue (thousands)

Fit a linear regression model to predict `Y` from `X1`, `X2`, `X3`, and `X4`.
Identify which predictors are statistically significant at the 5% level.

Save your results to `/output/analysis.json` with this structure:

{
  "significant_predictors": ["X1", "X3"],
  "coefficients": {"X1": 2.3, "X2": 0.1, "X3": 1.8, "X4": 0.05},
  "p_values": {"X1": 0.001, "X2": 0.45, "X3": 0.003, "X4": 0.72},
  "r_squared": 0.65
}

List only predictors with p < 0.05 in `significant_predictors`. Include all
four predictors in `coefficients` and `p_values` regardless of significance.
