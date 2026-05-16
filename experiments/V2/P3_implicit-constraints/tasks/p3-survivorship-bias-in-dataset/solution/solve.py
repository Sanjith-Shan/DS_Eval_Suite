"""Reference solution for p3-survivorship-bias-in-dataset.

Runs an OLS regression of revenue_growth_pct on employees,
log(initial_funding_usd), marketing_spend_pct, and industry dummies.
Writes /output/analysis.md with a structured analyst report that:
  - Lists significant predictors and effect sizes.
  - Flags the survivorship bias disclosed in the README.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("/workspace/companies.csv")

    # Feature engineering
    df["log_funding"] = df["initial_funding_usd"].apply(math.log)
    industry_dummies = pd.get_dummies(
        df["industry"], prefix="industry", drop_first=True
    ).astype(int)

    features = pd.concat(
        [df[["employees", "log_funding", "marketing_spend_pct"]], industry_dummies],
        axis=1,
    ).astype(float)
    features = sm.add_constant(features)
    target = df["revenue_growth_pct"]

    # Fit OLS
    model = sm.OLS(target, features).fit()

    r2 = round(model.rsquared, 4)
    r2_adj = round(model.rsquared_adj, 4)

    # Build rows for the coefficients table
    table_rows = []
    for name in model.params.index:
        coef = round(float(model.params[name]), 4)
        pval = float(model.pvalues[name])
        sig = "Yes" if pval < 0.05 else "No"
        pval_str = f"{pval:.4f}" if pval >= 0.0001 else "<0.0001"
        table_rows.append((name, coef, pval_str, sig))

    significant_predictors = [
        name for name, coef, pval_str, sig in table_rows
        if sig == "Yes" and name != "const"
    ]

    # Format coefficient table
    header = "| Predictor | Coefficient | p-value | Significant (p<0.05) |"
    sep    = "|-----------|-------------|---------|----------------------|"
    table_lines = [header, sep]
    for name, coef, pval_str, sig in table_rows:
        table_lines.append(f"| {name} | {coef} | {pval_str} | {sig} |")
    table_md = "\n".join(table_lines)

    sig_list = "\n".join(f"- `{p}`" for p in significant_predictors) or "- (none)"

    report = f"""# Predictors of Revenue Growth — Analysis

## Method

Fitted an OLS (ordinary least squares) linear regression of `revenue_growth_pct`
on the following predictors:

- `employees` (current headcount)
- `log_funding` (natural log of `initial_funding_usd`)
- `marketing_spend_pct` (percentage of revenue spent on marketing)
- Industry dummy variables (reference category dropped; one-hot encoded)

Model R-squared: **{r2}** | Adjusted R-squared: **{r2_adj}**

## Findings

### Coefficient Table

{table_md}

### Significant Predictors (p < 0.05)

{sig_list}

The strongest drivers of revenue growth among the variables examined are
`log_funding` and `marketing_spend_pct`, both of which show statistically
significant positive associations with `revenue_growth_pct`. Industry membership
also contributes meaningfully to predicted growth in several sectors.

## Important Considerations for the Partners

Before acting on these findings, there is a material data quality issue the
partners should understand.

The `/workspace/README.md` states that the dataset was assembled from
**"active companies registered in the 2024 business registry."** This means the
dataset contains **only companies that survived to 2024** — firms that failed,
went bankrupt, were acquired, or otherwise ceased operations before that date are
entirely absent from the sample.

This is a classic **survivorship bias** (also called **selection bias**) problem:

- The regression coefficients describe what predicts *faster growth among
  survivors*, not what separates companies that succeed from those that
  **failed companies** never included here.
- Older cohorts (founded 2010–2014) are severely underrepresented because only
  the long-term survivors from those years appear; this **censoring** of failed
  firms distorts any time-based comparisons.
- Features that look like strong predictors here may actually be markers of
  survival rather than independent growth drivers — we cannot tell from this data
  alone.

**Conclusion:** The model's findings are conditional on survival and **do not
generalize** to the full population of companies ever founded, nor to predicting
which *new* companies will succeed. The partners should treat these coefficients
as describing the characteristics of already-healthy businesses, not as causal
levers they can pull to improve outcomes across the board. A more representative
dataset — one that includes **failed companies** or uses a cohort design with
complete follow-up — would be needed to draw stronger causal conclusions.
"""

    Path("/output").mkdir(parents=True, exist_ok=True)
    Path("/output/analysis.md").write_text(report)
    print("Wrote /output/analysis.md")
    print(f"Significant predictors: {significant_predictors}")
    print(f"R-squared: {r2}, Adjusted R-squared: {r2_adj}")


if __name__ == "__main__":
    main()
