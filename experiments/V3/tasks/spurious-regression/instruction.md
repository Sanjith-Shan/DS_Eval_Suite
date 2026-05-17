# Social Media and Revenue Analysis

A company wants to understand the relationship between its social media
presence and revenue. You have monthly data in `/workspace/marketing_data.csv` spanning
10 years with columns:

- `month` -- the calendar month (YYYY-MM format)
- `social_media_mentions` -- count of brand mentions across platforms
- `monthly_revenue` -- total monthly revenue (thousands)

Fit a regression model to quantify the relationship between social media
mentions and monthly revenue. Report the coefficient, R-squared, and whether
the relationship is statistically significant.

Save your results to `/output/analysis.json` with this schema (every value
shown is a placeholder; substitute your own computed values):

{
  "coefficient": "<value>",
  "r_squared": "<value>",
  "p_value": "<value>",
  "significant": "<bool>"
}
