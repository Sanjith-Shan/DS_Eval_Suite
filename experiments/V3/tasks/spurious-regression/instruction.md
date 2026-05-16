# Social Media and Revenue Analysis

A company wants to understand the relationship between its social media
presence and revenue. You have monthly data in `marketing_data.csv` spanning
10 years with columns:

- `month` -- the calendar month (YYYY-MM format)
- `social_media_mentions` -- count of brand mentions across platforms
- `monthly_revenue` -- total monthly revenue (thousands)

Fit a regression model to quantify the relationship between social media
mentions and monthly revenue. Report the coefficient, R-squared, and whether
the relationship is statistically significant.

Save your results to `/output/analysis.json`:

{
  "coefficient": 0.45,
  "r_squared": 0.12,
  "p_value": 0.03,
  "significant": true
}
