# Advertising and Sales Analysis

You have data on advertising spend and sales revenue for 200 observations
in `advertising_data.csv`. The columns are:

- `observation_id` -- unique identifier
- `advertising_spend` -- monthly advertising budget (thousands)
- `sales_revenue` -- monthly sales revenue (thousands)

Fit a linear regression of `sales_revenue` on `advertising_spend`. Report
the regression coefficient, R-squared, p-value, and the number of
observations used in the final model.

Save your results to `/output/analysis.json`:

{
  "coefficient": 0.45,
  "r_squared": 0.12,
  "p_value": 0.001,
  "significant": true,
  "n_observations_used": 196
}
