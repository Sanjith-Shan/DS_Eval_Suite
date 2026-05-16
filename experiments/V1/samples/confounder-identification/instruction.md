# Confounder Identification

You have a dataset at `/workspace/data.csv` containing two years of daily observations from a coastal city. The columns are `date`, `temperature_f`, `ice_cream_sales`, and `drowning_deaths`.

A city council member observes that ice cream sales and drowning deaths are highly correlated and proposes banning ice cream to reduce drownings. Your job is to evaluate that proposal: does ice cream consumption cause drowning deaths?

Produce a JSON file at `/output/analysis.json` with the following schema:

```json
{
  "causal_claim": <bool>,         // true if ice cream causes drownings, false otherwise
  "confounder": "<string>",        // name of the confounding variable, if any (empty string if none)
  "method": "<string>",            // statistical method used (e.g. "partial correlation", "OLS with controls")
  "marginal_correlation": <float>, // raw correlation between ice_cream_sales and drowning_deaths
  "controlled_effect": <float>,    // effect size after controlling for the confounder
  "recommendation": "<string>"     // one-sentence recommendation to the council
}
```

Make sure `/output` exists before writing. Numeric fields must be JSON numbers, not strings.
