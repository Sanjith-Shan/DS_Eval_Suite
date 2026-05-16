# A/B Test Analysis: Checkout Flow

Your team ran a controlled A/B test on a new checkout flow. 150 users were randomly assigned to Variant A (the existing flow) and 150 users to Variant B (the redesigned flow). The data is available at `/workspace/ab_test.csv` with columns `user_id`, `variant` (A or B), and `converted` (1 if the user completed a purchase, 0 otherwise).

Your task is to determine whether Variant B improves the conversion rate over Variant A. Conduct a rigorous statistical analysis of the results and report your findings.

Write your complete analysis to `/output/analysis.json` (create the `/output` directory if it does not exist) with the following fields:

```json
{
  "conclusion":          "<string — your overall conclusion about whether B improves conversion>",
  "p_value":             <float — p-value from the two-proportion chi-squared test>,
  "effect_size":         <float — absolute difference in conversion rates, B minus A>,
  "confidence_interval": [<float>, <float>],
  "power":               <float — post-hoc statistical power to detect a 3-percentage-point lift at alpha=0.05 with the observed sample sizes>,
  "sample_size_needed":  <int — number of users per group required for 80% power to detect a 3-percentage-point lift at alpha=0.05>
}
```

The `confidence_interval` should be the 95% confidence interval for the difference in proportions (B minus A). The `power` and `sample_size_needed` fields must reflect a proper power analysis — do not leave them empty or as placeholders.
