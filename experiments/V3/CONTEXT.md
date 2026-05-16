# CONTEXT.md -- Assumption Verification Eval Suite

## What This Project Is

This is a take-home assignment for Abundant.AI. We are building 10 Harbor-format evaluation tasks that test whether Gemini Flash can independently verify statistical assumptions. All 10 tasks must target < 30% pass@3 against `gemini-3-flash-preview`.

4 tasks are already built and confirmed (0/3 against Gemini). 6 tasks are new and need to be built from scratch, then all 10 tested.

The Harbor reference docs and usage instructions are provided separately.

---

## Project Structure

```
DS_Eval_Suite/
├── CONTEXT.md                          # this file
├── samples/                            # final submission folder (all 10 tasks)
│   ├── multicollinearity-after-log/
│   ├── longitudinal-data-structure/
│   ├── clustered-parametric-test/
│   ├── survivorship-bias-sample/
│   ├── autocorrelated-residuals/
│   ├── multiple-comparisons/
│   ├── influential-outliers/
│   ├── clustered-treatment/
│   ├── censored-survival/
│   └── spurious-regression/
├── logs/                               # harbor run output (>= 3 trials per task)
└── report/                             # final report
```

Each task folder follows this exact layout:
```
<task-name>/
├── instruction.md
├── task.toml
├── environment/
│   ├── Dockerfile
│   └── <data>.csv
├── tests/
│   ├── test.sh
│   └── verify.py
└── solution/
    ├── solve.sh
    └── solve.py
```

---

## Shared Technical Specs (apply to ALL 10 tasks)

### task.toml template

```toml
schema_version = "1.2"
name = "sanjith/<task-name>"
version = "1.0.0"
description = "<one-line description>"
authors = [{ name = "Sanjith" }]
allow_internet = true

[environment]
dockerfile = "environment/Dockerfile"

[environment.env]

[verifier]
dockerfile = "environment/Dockerfile"

[verifier.env]

[solution]
dockerfile = "environment/Dockerfile"

[solution.env]
```

### Dockerfile template

```dockerfile
FROM python:3.11-slim

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scipy==1.13.1 \
    scikit-learn==1.5.1 \
    statsmodels==0.14.2

WORKDIR /workspace
COPY *.csv /workspace/
```

For task N5 (censored-survival), add `lifelines==0.29.0` to the pip install line.

### test.sh template

Every test.sh follows this exact pattern:

```bash
#!/bin/bash
set -e
cd /workspace
python /tasks/tests/verify.py
```

The verify.py script must handle all logic and write the reward:

```python
import json, sys, os

def main():
    reward_dir = "/logs/verifier"
    os.makedirs(reward_dir, exist_ok=True)
    reward_path = os.path.join(reward_dir, "reward.txt")
    
    try:
        # ... verification logic ...
        # If all checks pass:
        with open(reward_path, "w") as f:
            f.write("1")
        print(f"reward=1 reason=all checks passed")
        return True
    except Exception as e:
        with open(reward_path, "w") as f:
            f.write("0")
        print(f"reward=0 reason={e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0)  # always exit 0; reward file carries the signal
```

### solve.sh template

```bash
#!/bin/bash
set -e
cd /workspace
python /tasks/solution/solve.py
```

### instruction.md rules

- 2-3 paragraphs max
- Describes a standard, realistic data science deliverable
- NEVER mentions the assumption that needs to be verified
- NEVER hints at the diagnostic step
- Specifies the output path and format (JSON unless noted)
- Written as if from a manager to a data scientist

### Verifier rules

- Checks numeric values or structural properties of the output, NEVER checks process/code
- No AST inspection, no keyword grep on agent code, no LLM-as-judge
- Deterministic -- same output always gets same reward
- Must have a large gap between "assumption verified" and "assumption not verified" outputs
- Always writes reward to /logs/verifier/reward.txt

### Data generation

- Every dataset has a standalone generate_data.py with a fixed seed
- The generate_data.py is NOT shipped in the task folder -- it exists only for reproducibility
- The generated CSV is placed in environment/ and ships with the task
- All data generation uses numpy/pandas with explicit random seeds

---

## EXISTING TASKS (4 tasks -- copy from prior experiments, adjust task.toml name only)

These 4 tasks are already fully built and confirmed at 0/3 against Gemini. They need to be copied into the samples/ folder. Their source locations are listed below. The ONLY change needed is updating the `name` field in task.toml to match the new naming convention.

### E1: multicollinearity-after-log

**Source:** `experiments/P1_surface-consequence/tasks/multicollinearity-after-log-transform/`
**New task.toml name:** `sanjith/multicollinearity-after-log`
**Gemini result:** 0/3 (identical failure, X2_pvalue=0.122 in every trial)

Summary of what this task does (for reference, do NOT rebuild):
- Agent is told to fix heteroscedasticity and report OLS coefficient p-values
- Log(Y) fixes the heteroscedasticity but pushes X1-X2 correlation to ~0.99
- Agent must independently check VIF, then use ridge regression with bootstrap p-values
- Verifier requires all 4 p-values < 0.05 AND Breusch-Pagan p > 0.05
- Plain OLS after log gives X2_pvalue=0.122 (fails); ridge gives X2_pvalue=0.032 (passes)

### E2: longitudinal-data-structure

**Source:** `experiments/P1_surface-consequence/tasks/deduplication-loses-valid-longitudinal-data/`
**New task.toml name:** `sanjith/longitudinal-data-structure`
**Gemini result:** 0/3 (AUC=0.6733 in every trial)

Summary:
- Agent is told to clean data (repeated patient_ids) and build a readmission classifier
- Repeated IDs are multi-visit longitudinal records, not duplicates
- Agent must recognize longitudinal structure and engineer delta features (current - previous visit)
- Verifier requires AUC >= 0.72
- Without delta features AUC=0.6733 (fails); with delta features AUC=0.884 (passes)

### E3: clustered-parametric-test

**Source:** `experiments/P2_cascading-multistep/tasks/wrong-sampling-cascades-to-wrong-test/`
**New task.toml name:** `sanjith/clustered-parametric-test`
**Gemini result:** 0/3 (all three used Welch's t-test)

Summary:
- 500 stores x 1000 rows of clustered transaction data
- Agent is told to sample, describe, pick a stat test, and report
- Clustered structure requires stratified sampling + non-parametric test (Mann-Whitney)
- Verifier requires test_used to match non-parametric vocabulary AND reject parametric vocabulary
- Gemini used Welch's t-test all 3 trials (fails)

### E4: survivorship-bias-sample

**Source:** `experiments/P3_implicit-constraints/tasks/p3-survivorship-bias-in-dataset/`
**New task.toml name:** `sanjith/survivorship-bias-sample`
**Gemini result:** 0/3 (three different failure modes, all on-pattern)

Summary:
- Dataset contains only companies that survived to 2024
- Agent is told to analyze predictors of company growth and write a report
- Agent must independently flag survivorship/selection bias in the report
- Verifier dual-gate: requires BOTH analysis content keywords AND bias-awareness keywords
- Gemini failed to flag bias in all 3 trials

---

## NEW TASKS (6 tasks -- build from scratch)

### N1: autocorrelated-residuals

**Assumption tested:** Independence of residuals (OLS assumption #4)

**The trap:** Monthly time-ordered data where OLS residuals are strongly autocorrelated (Durbin-Watson ~0.6). Autocorrelation inflates standard errors for some predictors, creating a false positive. One predictor (X2) appears significant under naive OLS (p ~0.03) but is NOT significant under Newey-West HAC standard errors (p ~0.15). The model must independently check residual autocorrelation and use robust standard errors.

#### generate_data.py specification

```
Seed: 42
N: 120 observations (monthly, Jan 2015 - Dec 2024)

True model:
  Y = 2.5*X1 + 0.0*X2 + 1.8*X3 + 0.0*X4 + epsilon
  where epsilon follows AR(1) with rho = 0.7 and innovation_sd = 3.0

Generate predictors:
  X1 ~ N(10, 2) with mild temporal trend (add 0.02*t)
  X2 ~ N(5, 1.5) with moderate temporal trend (add 0.03*t) 
  X3 ~ N(8, 2.5) iid
  X4 ~ N(3, 1) iid

The temporal trends in X1 and X2 combined with the AR(1) errors create 
correlation between X2 and the residual process, which inflates X2's 
t-statistic under naive OLS.

Generate AR(1) noise:
  e[0] ~ N(0, innovation_sd)
  e[t] = rho * e[t-1] + N(0, innovation_sd)

Y = 2.5*X1 + 0.0*X2 + 1.8*X3 + 0.0*X4 + e

Output columns: month, X1, X2, X3, X4, Y
month format: "2015-01", "2015-02", ..., "2024-12"

VALIDATION (run after generation, assert all hold):
  1. Naive OLS p-value for X2 < 0.05 (false positive present)
  2. Naive OLS p-value for X1 < 0.05 (true positive)
  3. Naive OLS p-value for X3 < 0.05 (true positive)
  4. Naive OLS p-value for X4 > 0.10 (true negative)
  5. Durbin-Watson statistic < 1.0 (strong autocorrelation)
  6. Newey-West p-value for X2 > 0.05 (X2 no longer significant)
  7. Newey-West p-value for X1 < 0.05 (X1 still significant)
  8. Newey-West p-value for X3 < 0.05 (X3 still significant)

If validation fails, adjust parameters (rho, trend slopes, innovation_sd) 
and re-run until all 8 conditions hold. The key tuning knobs are:
  - rho (higher = stronger autocorrelation = more SE inflation)
  - X2 trend slope (higher = more spurious correlation with AR errors)
  - innovation_sd (higher = noisier, makes marginal predictors less significant)

Save as: environment/business_data.csv
```

#### instruction.md

```markdown
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
```

#### verify.py logic

```
1. Load /output/analysis.json
2. Check required fields: significant_predictors, coefficients, p_values, r_squared
3. Check "X1" IS in significant_predictors (true positive must be found)
4. Check "X3" IS in significant_predictors (true positive must be found)
5. Check "X2" is NOT in significant_predictors  <-- THIS IS THE KEY DISCRIMINATOR
6. Check "X4" is NOT in significant_predictors (true negative)
7. Check r_squared > 0 and r_squared < 1

If all pass: reward = 1
If any fail: reward = 0, print which check failed and the actual values
```

The critical check is #5. Naive OLS gives X2 p ~0.03 (would put it in significant_predictors). Newey-West gives X2 p ~0.15 (would exclude it). An agent that doesn't check for autocorrelation will include X2 and fail.

#### solve.py (oracle)

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
import json, os

def main():
    df = pd.read_csv("business_data.csv")
    X = df[["X1", "X2", "X3", "X4"]]
    y = df["Y"]
    X_const = sm.add_constant(X)
    
    # Step 1: Fit OLS
    ols = sm.OLS(y, X_const).fit()
    
    # Step 2: Check Durbin-Watson (THE ASSUMPTION VERIFICATION STEP)
    dw = durbin_watson(ols.resid)
    print(f"Durbin-Watson: {dw:.3f}")
    # dw should be ~0.6, indicating strong positive autocorrelation
    
    # Step 3: Refit with Newey-West HAC standard errors
    robust = ols.get_robustcov_results(cov_type="HAC", maxlags=int(np.ceil(len(df)**0.25)))
    
    # Step 4: Report based on robust p-values
    predictors = ["X1", "X2", "X3", "X4"]
    coefficients = {p: float(robust.params[p]) for p in predictors}
    p_values = {p: float(robust.pvalues[p]) for p in predictors}
    significant = [p for p in predictors if p_values[p] < 0.05]
    
    result = {
        "significant_predictors": significant,
        "coefficients": coefficients,
        "p_values": p_values,
        "r_squared": float(ols.rsquared)
    }
    
    os.makedirs("/output", exist_ok=True)
    with open("/output/analysis.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Significant (robust): {significant}")
    print(f"P-values (robust): {p_values}")

if __name__ == "__main__":
    main()
```

---

### N2: multiple-comparisons

**Assumption tested:** Family-wise error rate control when running many tests

**The trap:** 8 drug formulations, 28 pairwise comparisons. At alpha=0.05 with 28 tests, expected false positives ~1.4. We engineer the data so that exactly 1 pair has a genuine effect and 2-3 pairs are false positives at 0.05 uncorrected. After Bonferroni/BH correction, only the real pair survives. The model must independently decide to correct for multiple comparisons.

#### generate_data.py specification

```
Seed: 42
8 groups (formulations A through H), 50 observations each = 400 total

True group means:
  A: 100    E: 100
  B: 100    F: 100  
  C: 100    G: 100
  D: 108    H: 100
  (Only D differs from the rest. Cohen's d for A-vs-D = 8/15 = 0.53)

Within-group SD: 15 for all groups

Generate: for each group, draw 50 values from N(mean, 15)

VALIDATION (run after generation, assert all hold):
  1. Run all 28 pairwise independent two-sample t-tests at alpha=0.05
  2. Count pairs with p < 0.05. Must be >= 3 (1 real + 2+ false positives)
  3. The real pair (any pair involving D vs a non-D group) must have p < 0.001
  4. Apply Bonferroni correction (alpha = 0.05/28 = 0.001786)
  5. After Bonferroni, the number of significant pairs must be <= 2
  6. After Bonferroni, at least one D-vs-other pair must remain significant

If validation fails, adjust the seed or the true mean for D until conditions hold.
The key tuning knob is the mean of D (higher = more real pairs survive Bonferroni, 
lower = real pair might not survive). 108 with SD=15 should work.

IMPORTANT: the false positives must be between non-D groups (e.g., B vs F, 
or A vs G). If the seed doesn't produce at least 2 false positives among 
the 21 non-D pairwise tests, try seeds 43, 44, etc. until it does. With 
21 tests at alpha=0.05, probability of >= 2 false positives is ~1 - (0.95^21 + 21*0.05*0.95^20) ≈ 0.66, so most seeds will work.

Output columns: patient_id, formulation, efficacy_score
Save as: environment/drug_trial.csv
```

#### instruction.md

```markdown
# Drug Formulation Comparison

A pharmaceutical company tested 8 formulations (A through H) of a new drug. 
Each formulation was given to 50 patients, and their efficacy scores were 
recorded in `drug_trial.csv`. The columns are:

- `patient_id` -- unique patient identifier
- `formulation` -- which formulation the patient received (A-H)
- `efficacy_score` -- measured treatment efficacy

For every pair of formulations, test whether their mean efficacy scores 
differ significantly at the 5% level. Report all pairs that show a 
statistically significant difference.

Save your results to `/output/analysis.json` with this structure:

{
  "significant_pairs": [["A", "D"], ["B", "F"]],
  "total_comparisons": 28,
  "test_used": "independent two-sample t-test",
  "details": {
    "A_vs_D": {"t_statistic": 3.2, "p_value": 0.002},
    "B_vs_F": {"t_statistic": 2.1, "p_value": 0.04}
  }
}

Each entry in `significant_pairs` should be a two-element list with the 
formulation letters in alphabetical order. Include test details for the 
significant pairs.
```

#### verify.py logic

```
1. Load /output/analysis.json
2. Check required fields: significant_pairs, total_comparisons
3. Normalize all pairs to alphabetical order for comparison
4. Check that at least one pair involving "D" is in significant_pairs 
   (the real signal must be detected)
5. Check len(significant_pairs) <= 2  <-- THIS IS THE KEY DISCRIMINATOR
   (Uncorrected results will have 3-4 pairs; corrected will have 1-2)
6. Check total_comparisons == 28 (or >= 20, to be lenient on methodology)

If all pass: reward = 1
If #5 fails (too many significant pairs): reward = 0, reason = "too many 
significant pairs reported ({n}), expected <= 2 after correction for 
multiple comparisons"
```

The critical check is #5. Without multiple comparison correction, 3-4 pairs will be reported as significant. With correction, only 1-2 survive. The verifier doesn't check WHETHER the agent applied Bonferroni -- it checks the CONSEQUENCE (number of reported pairs).

#### solve.py (oracle)

```python
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import json, os

def main():
    df = pd.read_csv("drug_trial.csv")
    formulations = sorted(df["formulation"].unique())
    
    pairs = list(combinations(formulations, 2))
    results = {}
    p_values = []
    
    for a, b in pairs:
        data_a = df[df["formulation"] == a]["efficacy_score"]
        data_b = df[df["formulation"] == b]["efficacy_score"]
        t_stat, p_val = stats.ttest_ind(data_a, data_b)
        key = f"{a}_vs_{b}"
        results[key] = {"t_statistic": float(t_stat), "p_value": float(p_val)}
        p_values.append((a, b, p_val))
    
    # Apply Bonferroni correction
    n_tests = len(pairs)
    alpha_corrected = 0.05 / n_tests
    
    significant = []
    details = {}
    for a, b, p in p_values:
        if p < alpha_corrected:
            significant.append([a, b])
            details[f"{a}_vs_{b}"] = results[f"{a}_vs_{b}"]
    
    output = {
        "significant_pairs": significant,
        "total_comparisons": n_tests,
        "test_used": f"independent two-sample t-test with Bonferroni correction (alpha={alpha_corrected:.6f})",
        "details": details
    }
    
    os.makedirs("/output", exist_ok=True)
    with open("/output/analysis.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Corrected alpha: {alpha_corrected:.6f}")
    print(f"Significant pairs: {significant}")

if __name__ == "__main__":
    main()
```

---

### N3: influential-outliers

**Assumption tested:** No unduly influential observations driving the regression

**The trap:** 200 observations with a weak positive relationship (true coefficient ~0.3). 4 high-leverage points at extreme X values with extreme Y values inflate the apparent coefficient to ~0.75. These points are NOT outliers in Y alone (their Y values are within 2 SD of the Y mean) -- they are only detectable via leverage/Cook's distance analysis. The model must independently check influence diagnostics and either remove or downweight the influential points.

#### generate_data.py specification

```
Seed: 42
N_clean: 196 observations
N_influential: 4 observations
Total: 200

Clean data:
  X ~ N(50, 10)
  Y = 0.30 * X + N(0, 8)  (weak positive relationship)
  
Influential points (high leverage, pulling slope up):
  X values: [90, 93, 96, 99]  (far from X mean of 50)
  Y values: [75, 80, 85, 90]  (these are high but within 2 SD of Y mean)
  
  These points have high leverage (extreme X) AND high residual 
  (Y is much higher than 0.30*X would predict), so Cook's distance 
  will be very high.

VALIDATION (run after generation, assert all hold):
  1. Full-data OLS coefficient > 0.60 (inflated by influential points)
  2. Clean-data OLS coefficient in [0.20, 0.45] (the true relationship)
  3. Full-data R² > 0.25
  4. Clean-data R² < 0.15
  5. All 4 influential points have Cook's distance > 4/N = 0.02
  6. No clean point has Cook's distance > 0.02
  7. The 4 influential points are NOT outliers by simple Y z-score 
     (|z_Y| < 2.5 for all 4)

If validation fails, adjust Y noise SD or influential point Y values.
Key tuning: the influential points need to be far enough in X to create 
leverage, with Y values high enough to pull the slope, but not so high 
in Y that simple outlier detection catches them.

Output columns: observation_id, advertising_spend, sales_revenue
(X = advertising_spend, Y = sales_revenue)
Save as: environment/advertising_data.csv
```

#### instruction.md

```markdown
# Advertising and Sales Analysis

You have data on advertising spend and sales revenue for 200 observations 
in `advertising_data.csv`. The columns are:

- `observation_id` -- unique identifier
- `advertising_spend` -- monthly advertising budget (thousands)
- `sales_revenue` -- monthly sales revenue (thousands)

Fit a linear regression of `sales_revenue` on `advertising_spend`. Report 
the regression coefficient, R-squared, p-value, and the number of 
observations used in the final model.

Save your results to `/output/analysis.json` with this structure:

{
  "coefficient": 0.45,
  "r_squared": 0.12,
  "p_value": 0.001,
  "significant": true,
  "n_observations_used": 196
}
```

#### verify.py logic

```
1. Load /output/analysis.json
2. Check required fields: coefficient, r_squared, p_value, significant, n_observations_used
3. Check coefficient is in [0.15, 0.50]  <-- KEY DISCRIMINATOR
   (Full-data coefficient ~0.75 fails; clean-data coefficient ~0.30 passes)
4. Check r_squared < 0.25 (clean-data R² is ~0.08-0.12)
5. Check significant is true (the relationship is real, just weaker)
6. Check n_observations_used <= 198 (must have removed at least 2 points)
   OR check coefficient is in band (primary check; n_observations is secondary)

If all pass: reward = 1
If coefficient > 0.50: reward = 0, reason = "coefficient {val} > 0.50, 
likely includes influential outliers (expected ~0.30 after removal)"
```

#### solve.py (oracle)

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import json, os

def main():
    df = pd.read_csv("advertising_data.csv")
    X = df[["advertising_spend"]]
    y = df["sales_revenue"]
    X_const = sm.add_constant(X)
    
    # Step 1: Fit initial OLS
    model = sm.OLS(y, X_const).fit()
    
    # Step 2: Compute Cook's distance (THE ASSUMPTION VERIFICATION STEP)
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    threshold = 4 / len(df)
    
    # Step 3: Remove influential points
    influential_mask = cooks_d > threshold
    n_removed = influential_mask.sum()
    print(f"Removing {n_removed} influential points (Cook's d > {threshold:.4f})")
    
    df_clean = df[~influential_mask]
    X_clean = df_clean[["advertising_spend"]]
    y_clean = df_clean["sales_revenue"]
    X_clean_const = sm.add_constant(X_clean)
    
    # Step 4: Refit on clean data
    model_clean = sm.OLS(y_clean, X_clean_const).fit()
    
    result = {
        "coefficient": float(model_clean.params["advertising_spend"]),
        "r_squared": float(model_clean.rsquared),
        "p_value": float(model_clean.pvalues["advertising_spend"]),
        "significant": bool(model_clean.pvalues["advertising_spend"] < 0.05),
        "n_observations_used": int(len(df_clean))
    }
    
    os.makedirs("/output", exist_ok=True)
    with open("/output/analysis.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Original coefficient: {model.params['advertising_spend']:.4f}")
    print(f"Clean coefficient: {model_clean.params['advertising_spend']:.4f}")

if __name__ == "__main__":
    main()
```

---

### N4: clustered-treatment

**Assumption tested:** Independence of observations when treatment is assigned at group level

**The trap:** An education study where a teaching method was assigned at the section level (not student level). 40 sections, 25 students each. Within-section correlation (ICC ~0.25) means student-level observations are NOT independent. A naive student-level t-test (n=1000) gives p ~0.002 (falsely significant). Aggregating to section means (n=40) gives p ~0.18 (correctly non-significant). The model must independently recognize that treatment varies at the section level and adjust the unit of analysis.

#### generate_data.py specification

```
Seed: 42
N_sections: 40 (20 treatment, 20 control)
Students_per_section: 25
Total students: 1000

Data generating process:
  Section-level effect: section_mean ~ N(75, 4) for control, N(76.5, 4) for treatment
  Student-level noise: score = section_mean + N(0, 7)
  
  This gives ICC = 4^2 / (4^2 + 7^2) = 16/65 ≈ 0.246
  
  True treatment effect at section level: 1.5 points (small, Cohen's d ≈ 1.5/4 = 0.375 at section level)
  But with only 20 sections per group and between-section SD=4, this is NOT significant
  
  At student level, the pooled SD ≈ sqrt(4^2 + 7^2) ≈ 8.06
  Student-level Cohen's d ≈ 1.5/8.06 ≈ 0.186 (very small)
  But n=500 per group makes even tiny effects "significant" in a naive t-test

VALIDATION (run after generation, assert all hold):
  1. Student-level t-test p-value < 0.01 (inflated significance)
  2. Section-level t-test on section means p-value > 0.05 (correct result)
  3. Section-level t-test p-value > 0.10 (clearly non-significant)
  4. ICC computed from one-way ANOVA is between 0.15 and 0.35

If validation fails, adjust section-level SD, student noise SD, or 
treatment effect size. Key constraint: student-level p must be < 0.01 
AND section-level p must be > 0.05. The treatment effect (1.5) and 
section SD (4) are the main tuning knobs.

Output columns: student_id, section_id, treatment_group, test_score
treatment_group values: "new_method" or "standard"
Save as: environment/education_study.csv
```

#### instruction.md

```markdown
# Teaching Method Evaluation

An education researcher conducted a study comparing a new teaching method 
against the standard approach. Students were organized into 40 sections, 
with each section using either the new method or the standard method. Test 
scores were recorded for all students.

The data is in `education_study.csv` with columns:

- `student_id` -- unique student identifier
- `section_id` -- which section the student belongs to
- `treatment_group` -- "new_method" or "standard"
- `test_score` -- the student's score on the final exam

Determine whether the new teaching method produces significantly better 
test scores than the standard method. Report your findings.

Save your results to `/output/analysis.json` with this structure:

{
  "significant": true,
  "p_value": 0.03,
  "effect_size": 2.5,
  "test_used": "independent two-sample t-test",
  "mean_new_method": 78.2,
  "mean_standard": 75.7
}
```

#### verify.py logic

```
1. Load /output/analysis.json
2. Check required fields: significant, p_value
3. PRIMARY CHECK: p_value must be > 0.05  <-- KEY DISCRIMINATOR
   (Student-level t-test gives ~0.002, section-level gives ~0.18)
4. OR: significant must be false
5. Either check is sufficient: if p_value > 0.05 OR significant == false, pass

If p_value <= 0.05 AND significant is true: reward = 0, 
  reason = "p_value={val} suggests student-level analysis without 
  accounting for section-level clustering (expected p > 0.05 after 
  proper aggregation)"
```

#### solve.py (oracle)

```python
import pandas as pd
import numpy as np
from scipy import stats
import json, os

def main():
    df = pd.read_csv("education_study.csv")
    
    # Step 1: Examine data structure
    print(f"Total students: {len(df)}")
    print(f"Sections: {df['section_id'].nunique()}")
    print(f"Students per section: {df.groupby('section_id').size().mean():.0f}")
    
    # Step 2: Check -- does treatment vary within sections?
    section_treatments = df.groupby("section_id")["treatment_group"].nunique()
    print(f"Treatments per section: {section_treatments.unique()}")
    # All sections have exactly 1 treatment -- treatment assigned at section level!
    
    # Step 3: Aggregate to section level (THE ASSUMPTION VERIFICATION STEP)
    section_means = df.groupby(["section_id", "treatment_group"])["test_score"].mean().reset_index()
    
    new_method = section_means[section_means["treatment_group"] == "new_method"]["test_score"]
    standard = section_means[section_means["treatment_group"] == "standard"]["test_score"]
    
    t_stat, p_val = stats.ttest_ind(new_method, standard)
    effect = new_method.mean() - standard.mean()
    
    result = {
        "significant": bool(p_val < 0.05),
        "p_value": float(p_val),
        "effect_size": float(effect),
        "test_used": "t-test on section means (aggregated to unit of treatment assignment)",
        "mean_new_method": float(new_method.mean()),
        "mean_standard": float(standard.mean())
    }
    
    os.makedirs("/output", exist_ok=True)
    with open("/output/analysis.json", "w") as f:
        json.dump(result, f, indent=2)
    
    # Also show what naive student-level analysis would give
    naive_new = df[df["treatment_group"] == "new_method"]["test_score"]
    naive_std = df[df["treatment_group"] == "standard"]["test_score"]
    naive_t, naive_p = stats.ttest_ind(naive_new, naive_std)
    print(f"Naive student-level p: {naive_p:.6f}")
    print(f"Section-level p: {p_val:.6f}")

if __name__ == "__main__":
    main()
```

---

### N5: censored-survival

**Assumption tested:** Complete observation -- standard descriptive stats assume all events are observed

**The trap:** Clinical trial data where 35% of patients are right-censored (study ended before recovery, or they withdrew). A column called `study_status` indicates whether the patient actually recovered or was censored, but the instruction doesn't mention censoring. Naive analysis (mean/median of observed recovery_days, t-test) treats censored times as complete observations, biasing the estimate downward and creating a false significant difference between two drugs. Proper survival analysis (Kaplan-Meier for medians, log-rank for comparison) gives the correct answer.

#### generate_data.py specification

```
Seed: 42
N: 500 patients (250 drug_A, 250 drug_B)

True recovery time distribution:
  Drug A: Weibull(shape=1.5, scale=70)  -- median ~58 days
  Drug B: Weibull(shape=1.5, scale=72)  -- median ~60 days
  (Very similar -- NO real difference between drugs)

Censoring mechanism:
  Censoring time: Uniform(30, 90) for each patient
  Observed time = min(true_recovery, censoring_time)
  study_status = "recovered" if true_recovery <= censoring_time
                 "ongoing_at_study_end" if true_recovery > censoring_time AND rand > 0.15
                 "withdrew" if true_recovery > censoring_time AND rand <= 0.15

  This creates ~35% censoring overall.

Key bias mechanism:
  Drug A gets MORE censoring than Drug B by design:
  Drug A censoring times: Uniform(25, 80)  -- more aggressive censoring
  Drug B censoring times: Uniform(35, 95)  -- less aggressive censoring
  
  This means Drug A's "recovered" patients are biased toward FAST recoverers
  (slow recoverers in Drug A were more likely to be censored).
  Naive mean of observed recovery_days for Drug A will be LOWER (faster)
  than Drug B, creating a false impression that Drug A is better.

VALIDATION (run after generation, assert all hold):
  1. Overall censoring rate between 30% and 40%
  2. Drug A censoring rate > Drug B censoring rate (by design)
  3. Naive t-test on recovery_days (ignoring censoring): p < 0.05 
     (Drug A looks significantly faster)
  4. Log-rank test (proper survival comparison): p > 0.10 
     (no real difference)
  5. Naive median recovery for Drug A < naive median for Drug B 
     (biased by differential censoring)
  6. Kaplan-Meier median for Drug A ≈ Kaplan-Meier median for Drug B 
     (within 5 days)

Output columns: patient_id, drug, recovery_days, study_status
drug values: "drug_A", "drug_B"
study_status values: "recovered", "ongoing_at_study_end", "withdrew"
Save as: environment/clinical_trial.csv
```

#### instruction.md

```markdown
# Clinical Trial Analysis

A clinical trial compared two drugs (Drug A and Drug B) for patient recovery. 
The data is in `clinical_trial.csv` with columns:

- `patient_id` -- unique patient identifier
- `drug` -- which drug the patient received ("drug_A" or "drug_B")
- `recovery_days` -- number of days until recovery
- `study_status` -- patient's status ("recovered", "ongoing_at_study_end", 
  or "withdrew")

Analyze the data and determine whether there is a significant difference 
in recovery times between the two drugs. Report the median recovery time 
for each drug and whether the difference is statistically significant.

Save your results to `/output/analysis.json` with this structure:

{
  "median_recovery_drug_a": 58.0,
  "median_recovery_drug_b": 62.0,
  "significant_difference": false,
  "p_value": 0.35,
  "test_used": "log-rank test"
}
```

#### verify.py logic

```
1. Load /output/analysis.json
2. Check required fields: significant_difference, p_value
3. PRIMARY CHECK: p_value must be > 0.05  <-- KEY DISCRIMINATOR
   (Naive t-test gives p ~0.02; log-rank gives p ~0.30)
4. OR: significant_difference must be false
5. SECONDARY CHECK: median_recovery_drug_a must be > 50 
   (Naive median is ~42; KM median is ~58. Catches downward bias.)

If p_value <= 0.05 AND significant_difference is true: reward = 0,
  reason = "p_value={val} suggests naive comparison without accounting 
  for censored observations (expected p > 0.05 with survival analysis)"
```

Note: The verifier should NOT require `lifelines` to be installed. It only checks the JSON output values. The oracle solution uses lifelines (or manual KM computation).

#### solve.py (oracle)

```python
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, statistics as lf_stats
import json, os

def main():
    df = pd.read_csv("clinical_trial.csv")
    
    # Step 1: Recognize censoring (THE ASSUMPTION VERIFICATION STEP)
    # study_status tells us which observations are complete
    df["event_observed"] = (df["study_status"] == "recovered").astype(int)
    
    censoring_rate = 1 - df["event_observed"].mean()
    print(f"Censoring rate: {censoring_rate:.1%}")
    
    # Step 2: Kaplan-Meier for median recovery times
    kmf = KaplanMeierFitter()
    
    drug_a = df[df["drug"] == "drug_A"]
    drug_b = df[df["drug"] == "drug_B"]
    
    kmf.fit(drug_a["recovery_days"], event_observed=drug_a["event_observed"], label="Drug A")
    median_a = kmf.median_survival_time_
    
    kmf.fit(drug_b["recovery_days"], event_observed=drug_b["event_observed"], label="Drug B")
    median_b = kmf.median_survival_time_
    
    # Step 3: Log-rank test
    results = lf_stats.logrank_test(
        drug_a["recovery_days"], drug_b["recovery_days"],
        event_observed_A=drug_a["event_observed"],
        event_observed_B=drug_b["event_observed"]
    )
    
    output = {
        "median_recovery_drug_a": float(median_a),
        "median_recovery_drug_b": float(median_b),
        "significant_difference": bool(results.p_value < 0.05),
        "p_value": float(results.p_value),
        "test_used": "log-rank test (accounting for right-censored observations)"
    }
    
    os.makedirs("/output", exist_ok=True)
    with open("/output/analysis.json", "w") as f:
        json.dump(output, f, indent=2)
    
    # Show naive comparison for contrast
    naive_t, naive_p = __import__("scipy").stats.ttest_ind(
        drug_a["recovery_days"], drug_b["recovery_days"]
    )
    print(f"Naive t-test p: {naive_p:.6f}")
    print(f"Log-rank p: {results.p_value:.6f}")
    print(f"Naive median A: {drug_a['recovery_days'].median():.1f}")
    print(f"KM median A: {median_a:.1f}")

if __name__ == "__main__":
    main()
```

---

### N6: spurious-regression

**Assumption tested:** Stationarity of variables in regression (no spurious regression from unit roots)

**The trap:** Two independent non-stationary time series (both random walks with drift). Naive OLS shows R² ~0.80 and a highly significant coefficient (p < 0.001) even though the variables are completely independent. This is the classic Granger-Newbold spurious regression problem. After differencing both variables, R² drops to ~0.01 and p > 0.50. The model must independently check for stationarity (ADF test) and difference the variables before regressing.

#### generate_data.py specification

```
Seed: 42
N: 120 months (Jan 2015 - Dec 2024)

Generate two INDEPENDENT random walks:
  X[0] = 1000, Y[0] = 500
  X[t] = X[t-1] + drift_x + N(0, sigma_x)
  Y[t] = Y[t-1] + drift_y + N(0, sigma_y)
  
  drift_x = 5.0  (X trends upward, ~social media mentions growing)
  drift_y = 3.0  (Y trends upward, ~revenue growing)
  sigma_x = 15.0
  sigma_y = 10.0

X and Y are generated from COMPLETELY SEPARATE random processes.
Any correlation between them is purely spurious (both trend upward).

VALIDATION (run after generation, assert all hold):
  1. Naive OLS of Y on X: R² > 0.60 (spurious correlation from shared trend)
  2. Naive OLS of Y on X: p-value < 0.001 (highly "significant")
  3. ADF test on X: p-value > 0.05 (non-stationary, fail to reject unit root)
  4. ADF test on Y: p-value > 0.05 (non-stationary)
  5. ADF test on diff(X): p-value < 0.05 (stationary after differencing)
  6. ADF test on diff(Y): p-value < 0.05 (stationary after differencing)
  7. OLS of diff(Y) on diff(X): R² < 0.10 (no real relationship)
  8. OLS of diff(Y) on diff(X): p-value > 0.10 (not significant)
  9. Correlation between X and Y (levels) > 0.70 (strong spurious correlation)
  10. Correlation between diff(X) and diff(Y) < 0.20 (no real correlation)

Output columns: month, social_media_mentions, monthly_revenue
month format: "2015-01", "2015-02", ..., "2024-12"
Save as: environment/marketing_data.csv
```

#### instruction.md

```markdown
# Social Media and Revenue Analysis

A company wants to understand the relationship between its social media 
presence and revenue. You have monthly data in `marketing_data.csv` spanning 
10 years with columns:

- `month` -- the calendar month (YYYY-MM format)
- `social_media_mentions` -- count of brand mentions across platforms
- `monthly_revenue` -- total monthly revenue (thousands)

Fit a regression model to quantify the relationship between social media 
mentions and monthly revenue. Report the coefficient, R-squared, and 
whether the relationship is statistically significant.

Save your results to `/output/analysis.json` with this structure:

{
  "coefficient": 0.45,
  "r_squared": 0.12,
  "p_value": 0.03,
  "significant": true
}
```

#### verify.py logic

```
1. Load /output/analysis.json
2. Check required fields: r_squared, p_value
3. PRIMARY CHECK: r_squared must be < 0.15  <-- KEY DISCRIMINATOR
   (Naive OLS on levels: R² ~0.80. OLS on differences: R² ~0.01-0.05)
4. SECONDARY CHECK: p_value must be > 0.05 OR significant must be false
   (Differenced regression will be non-significant)

If r_squared >= 0.15: reward = 0,
  reason = "r_squared={val} suggests regression on non-stationary levels 
  (spurious regression). Expected R² < 0.15 after differencing."
```

The R² gap is enormous (0.80 vs 0.01), making this verifier very robust. Even if the model uses a different approach (log differences, detrending), the R² will be well below 0.15 if it addresses non-stationarity.

#### solve.py (oracle)

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import json, os

def main():
    df = pd.read_csv("marketing_data.csv")
    
    X = df["social_media_mentions"]
    Y = df["monthly_revenue"]
    
    # Step 1: Check stationarity (THE ASSUMPTION VERIFICATION STEP)
    adf_x = adfuller(X)
    adf_y = adfuller(Y)
    print(f"ADF test X: stat={adf_x[0]:.3f}, p={adf_x[1]:.4f}")
    print(f"ADF test Y: stat={adf_y[0]:.3f}, p={adf_y[1]:.4f}")
    # Both should have p > 0.05 (non-stationary)
    
    # Step 2: Difference both series
    dX = X.diff().dropna()
    dY = Y.diff().dropna()
    
    # Verify stationarity of differenced series
    adf_dx = adfuller(dX)
    adf_dy = adfuller(dY)
    print(f"ADF test diff(X): stat={adf_dx[0]:.3f}, p={adf_dx[1]:.4f}")
    print(f"ADF test diff(Y): stat={adf_dy[0]:.3f}, p={adf_dy[1]:.4f}")
    
    # Step 3: Regress differenced series
    dX_const = sm.add_constant(dX)
    model = sm.OLS(dY, dX_const).fit()
    
    result = {
        "coefficient": float(model.params.iloc[1]),
        "r_squared": float(model.rsquared),
        "p_value": float(model.pvalues.iloc[1]),
        "significant": bool(model.pvalues.iloc[1] < 0.05)
    }
    
    os.makedirs("/output", exist_ok=True)
    with open("/output/analysis.json", "w") as f:
        json.dump(result, f, indent=2)
    
    # Show naive result for contrast
    naive_model = sm.OLS(Y, sm.add_constant(X)).fit()
    print(f"Naive R²: {naive_model.rsquared:.4f}, p: {naive_model.pvalues.iloc[1]:.6f}")
    print(f"Differenced R²: {model.rsquared:.4f}, p: {model.pvalues.iloc[1]:.6f}")

if __name__ == "__main__":
    main()
```

---

## Build Checklist

For each of the 6 NEW tasks (N1-N6):

- [ ] Write generate_data.py with fixed seed
- [ ] Run generate_data.py, verify all validation conditions hold
- [ ] If validation fails, adjust parameters and re-run
- [ ] Save generated CSV to environment/
- [ ] Write instruction.md (no assumption hints)
- [ ] Write task.toml (schema 1.2, allow_internet = true)
- [ ] Write Dockerfile (python:3.11-slim + pinned deps)
- [ ] Write solution/solve.py (oracle)
- [ ] Write solution/solve.sh
- [ ] Write tests/verify.py
- [ ] Write tests/test.sh
- [ ] Local test: run solve.py, then verify.py -- must get reward=1
- [ ] Local test: run verify.py without solve.py output -- must get reward=0
- [ ] Local test: write a "naive" stub (plain OLS / uncorrected test), verify it gets reward=0

For the 4 EXISTING tasks (E1-E4):

- [ ] Copy from source location to samples/ folder
- [ ] Update task.toml name field
- [ ] Verify folder structure matches Harbor layout

Harbor testing (all 10 tasks):

- [ ] harbor run -a oracle for each task -- all must return reward=1
- [ ] harbor run -a nop for each task -- all must return reward=0
- [ ] harbor run -a gemini-cli -m google/gemini-3-flash-preview -k 3 for each task
- [ ] Record pass@1 and pass@3 for each task

---

## Parameter Tuning Guide

If a generate_data.py validation fails, here is how to adjust each task:

**N1 (autocorrelated-residuals):**
- If X2 is not a false positive under naive OLS: increase rho (AR coefficient) or increase X2 trend slope
- If X2 is still significant under Newey-West: decrease X2 trend slope or increase innovation_sd
- If DW is not < 1.0: increase rho

**N2 (multiple-comparisons):**
- If fewer than 3 uncorrected significant pairs: try different seeds (43, 44, ...) or decrease within-group SD
- If the real pair (involving D) doesn't survive Bonferroni: increase D's mean (try 110, 112)
- If too many pairs survive Bonferroni: decrease D's mean (try 106)

**N3 (influential-outliers):**
- If full-data coefficient is not > 0.60: move influential X values further out (try 95, 98, 101, 104) or increase their Y values
- If clean-data coefficient is not in [0.20, 0.45]: adjust the true slope or noise SD
- If influential points are Y-outliers (|z| > 2.5): lower their Y values slightly

**N4 (clustered-treatment):**
- If student-level p is not < 0.01: increase treatment effect (try 2.0) or decrease student noise
- If section-level p is not > 0.05: decrease treatment effect (try 1.0) or increase section SD
- Main tradeoff: treatment effect must be large enough to be "significant" at student level but not at section level

**N5 (censored-survival):**
- If naive t-test p is not < 0.05: increase differential censoring (make Drug A censoring more aggressive)
- If log-rank p is not > 0.10: make the true distributions more similar (closer scale parameters)
- If censoring rate is not 30-40%: adjust censoring time distributions

**N6 (spurious-regression):**
- If naive R² is not > 0.60: increase drift values or decrease noise SDs
- If differenced R² is not < 0.10: verify the series are truly independent (no shared noise component)
- If ADF tests don't show unit roots: increase noise SDs relative to drift
