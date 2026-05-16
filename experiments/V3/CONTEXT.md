# CONTEXT.md -- V3 Build Instructions

## Goal

Build 10 Harbor tasks testing statistical assumption verification. All must target < 30% pass@3 against `gemini-3-flash-preview`. 4 tasks exist in V2 (copy them in). 6 are new (build from scratch).

Every task: instruction asks for a standard analysis, data violates an assumption the instruction never mentions, model must independently catch and adjust. Verifier checks the numeric consequence, never the process.

Harbor docs are in `harbor-reference.md` (same directory as this file).

---

## V3 Layout

You are working inside `experiments/V3/`. Build this structure:

```
V3/
├── CONTEXT.md                              # this file (already exists)
├── harbor-reference.md                     # already exists
├── Abundant Research Take Home.pdf         # already exists
├── tasks/
│   ├── multicollinearity-after-log/        # E1 (copied from V2)
│   ├── longitudinal-data-structure/        # E2 (copied from V2)
│   ├── clustered-parametric-test/          # E3 (copied from V2)
│   ├── survivorship-bias-sample/           # E4 (copied from V2)
│   ├── autocorrelated-residuals/           # N1 (new)
│   ├── multiple-comparisons/              # N2 (new)
│   ├── influential-outliers/              # N3 (new)
│   ├── clustered-treatment/               # N4 (new)
│   ├── censored-survival/                 # N5 (new)
│   └── spurious-regression/               # N6 (new)
├── _build/
│   └── generate_<task>.py                  # one data generator per new task
├── jobs/                                   # harbor run outputs land here
└── REPORT.md
```

Each task folder follows Harbor layout:
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

## Shared Templates

### task.toml

```toml
schema_version = "1.2"
name = "sanjith/<task-name>"
version = "1.0.0"
description = "<one line>"
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

### Dockerfile (default)

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

For N5 (censored-survival), add `lifelines==0.29.0` to pip install.

### test.sh (all tasks use this exactly)

```bash
#!/bin/bash
set -e
cd /workspace
python /tasks/tests/verify.py
```

### solve.sh (all tasks use this exactly)

```bash
#!/bin/bash
set -e
cd /workspace
python /tasks/solution/solve.py
```

### verify.py skeleton

```python
import json, sys, os

def main():
    reward_dir = "/logs/verifier"
    os.makedirs(reward_dir, exist_ok=True)
    reward_path = os.path.join(reward_dir, "reward.txt")

    try:
        # load output, run checks, raise on failure
        with open(reward_path, "w") as f:
            f.write("1")
        print("reward=1 reason=all checks passed")
        return True
    except Exception as e:
        with open(reward_path, "w") as f:
            f.write("0")
        print(f"reward=0 reason={e}")
        return False

if __name__ == "__main__":
    main()
    sys.exit(0)  # always exit 0; reward file carries the signal
```

### instruction.md rules

- 2-3 paragraphs, written as a manager assigning work to a data scientist
- NEVER mentions the assumption, the diagnostic step, or hints at what's wrong with the data
- Specifies the output path (`/output/analysis.json`) and JSON schema

---

## EXISTING TASKS -- copy from V2

These 4 tasks are confirmed 0/3 against Gemini. Copy each folder into `tasks/`, rename the folder, and update the `name` field in task.toml. Change nothing else.

### E1: multicollinearity-after-log

**Copy from:** `../V2/P1_surface-consequence/tasks/multicollinearity-after-log-transform/`
**Rename folder to:** `multicollinearity-after-log`
**Set task.toml name to:** `sanjith/multicollinearity-after-log`

What it tests: log(Y) fixes heteroscedasticity but inflates X1-X2 correlation to ~0.99. Model must check VIF and use ridge regression. Verifier requires all 4 p-values < 0.05 AND Breusch-Pagan p > 0.05. Naive OLS after log gives X2_p=0.122 (fails).

### E2: longitudinal-data-structure

**Copy from:** `../V2/P1_surface-consequence/tasks/deduplication-loses-valid-longitudinal-data/`
**Rename folder to:** `longitudinal-data-structure`
**Set task.toml name to:** `sanjith/longitudinal-data-structure`

What it tests: repeated patient_ids are multi-visit longitudinal records. Model must engineer delta features. Verifier requires AUC >= 0.72. Without deltas AUC=0.6733 (fails).

### E3: clustered-parametric-test

**Copy from:** `../V2/P2_cascading-multistep/tasks/wrong-sampling-cascades-to-wrong-test/`
**Rename folder to:** `clustered-parametric-test`
**Set task.toml name to:** `sanjith/clustered-parametric-test`

What it tests: 500 stores x 1000 rows of clustered data. Model must use non-parametric test. Verifier rejects any test_used containing "t-test"/"welch"/"student". Gemini used Welch's t-test all 3 trials.

### E4: survivorship-bias-sample

**Copy from:** `../V2/P3_implicit-constraints/tasks/p3-survivorship-bias-in-dataset/`
**Rename folder to:** `survivorship-bias-sample`
**Set task.toml name to:** `sanjith/survivorship-bias-sample`

What it tests: dataset only contains companies surviving to 2024. Model must flag survivorship bias in its analysis report. Verifier dual-gate requires BOTH analysis keywords AND bias-awareness keywords.

---

## NEW TASKS -- build from scratch

For each new task, write a `_build/generate_<name>.py` that creates the CSV with a fixed seed. Run the generator, confirm the validation conditions hold, then place the CSV in `tasks/<name>/environment/`. The generator is NOT part of the task folder.

---

### N1: autocorrelated-residuals

**Assumption tested:** Independence of OLS residuals

**Trap:** 120 monthly observations. AR(1) residuals with rho=0.7 inflate standard errors. X2 appears significant under naive OLS (p ~0.03) but not under Newey-West HAC (p ~0.15). Model must check Durbin-Watson, then use robust SEs.

**Data generation (`_build/generate_autocorrelated.py`):**

```
Seed: 42
N: 120 (monthly, "2015-01" through "2024-12")

Predictors:
  X1 ~ N(10, 2) + 0.02*t   (true predictor, trending)
  X2 ~ N(5, 1.5) + 0.03*t  (null predictor, trending -- creates spurious correlation with AR errors)
  X3 ~ N(8, 2.5) iid        (true predictor)
  X4 ~ N(3, 1) iid           (null predictor)

AR(1) noise:
  e[0] ~ N(0, 3.0)
  e[t] = 0.7 * e[t-1] + N(0, 3.0)

Y = 2.5*X1 + 0.0*X2 + 1.8*X3 + 0.0*X4 + e

Columns: month, X1, X2, X3, X4, Y
File: environment/business_data.csv

VALIDATION (all must hold):
  1. Naive OLS p(X1) < 0.05
  2. Naive OLS p(X2) < 0.05   ← false positive
  3. Naive OLS p(X3) < 0.05
  4. Naive OLS p(X4) > 0.10
  5. Durbin-Watson < 1.0
  6. Newey-West p(X1) < 0.05
  7. Newey-West p(X2) > 0.05  ← corrected
  8. Newey-West p(X3) < 0.05

Tuning if validation fails:
  - X2 not false-positive? Increase rho or X2 trend slope
  - X2 still significant after Newey-West? Decrease X2 trend slope or increase innovation SD
```

**instruction.md:**

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

**verify.py checks:**

```
1. Load /output/analysis.json
2. Required fields: significant_predictors, coefficients, p_values, r_squared
3. "X1" in significant_predictors        (true positive)
4. "X3" in significant_predictors        (true positive)
5. "X2" NOT in significant_predictors    ← KEY DISCRIMINATOR
6. "X4" NOT in significant_predictors    (true negative)
7. r_squared > 0 and < 1
```

Naive OLS puts X2 in significant_predictors (p ~0.03) and fails check 5. Newey-West excludes X2 (p ~0.15) and passes.

**solve.py (oracle):**

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

    # Fit OLS
    ols = sm.OLS(y, X_const).fit()

    # Check Durbin-Watson
    dw = durbin_watson(ols.resid)
    print(f"Durbin-Watson: {dw:.3f}")

    # Refit with HAC standard errors
    robust = ols.get_robustcov_results(cov_type="HAC", maxlags=int(np.ceil(len(df)**0.25)))

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

if __name__ == "__main__":
    main()
```

---

### N2: multiple-comparisons

**Assumption tested:** Family-wise error rate control

**Trap:** 8 drug formulations, 28 pairwise t-tests. 7 groups share the same true mean, 1 group (D) is genuinely different. At alpha=0.05 uncorrected, 3-4 pairs appear significant (1 real + 2-3 false positives). After Bonferroni/BH, only the real pair(s) involving D survive. Model must independently apply multiple comparison correction.

**Data generation (`_build/generate_comparisons.py`):**

```
Seed: 42
8 groups (A-H), 50 per group = 400 total

Means: A=100, B=100, C=100, D=108, E=100, F=100, G=100, H=100
Within-group SD: 15

Generate 50 draws from N(mean, 15) per group.

Columns: patient_id, formulation, efficacy_score
File: environment/drug_trial.csv

VALIDATION:
  1. Uncorrected pairwise t-tests at alpha=0.05 yield >= 3 significant pairs
  2. At least one D-vs-other pair has p < 0.001
  3. After Bonferroni (alpha=0.05/28), significant pairs <= 2
  4. At least one D-vs-other pair survives Bonferroni

Tuning: if < 3 uncorrected false positives, try seeds 43, 44, ...
If real pair doesn't survive Bonferroni, increase D mean to 110.
```

**instruction.md:**

```markdown
# Drug Formulation Comparison

A pharmaceutical company tested 8 formulations (A through H) of a new drug.
Each formulation was given to 50 patients, and their efficacy scores were
recorded in `drug_trial.csv`. The columns are:

- `patient_id` -- unique patient identifier
- `formulation` -- which formulation (A-H)
- `efficacy_score` -- measured treatment efficacy

For every pair of formulations, test whether their mean efficacy scores differ
significantly at the 5% level. Report all pairs with significant differences.

Save your results to `/output/analysis.json`:

{
  "significant_pairs": [["A", "D"]],
  "total_comparisons": 28,
  "test_used": "t-test",
  "details": {
    "A_vs_D": {"t_statistic": 3.2, "p_value": 0.002}
  }
}

Each pair in `significant_pairs` should be alphabetically ordered.
```

**verify.py checks:**

```
1. Load /output/analysis.json
2. Required: significant_pairs, total_comparisons
3. Normalize pairs to alphabetical order
4. At least one pair involving "D" is present   (real signal detected)
5. len(significant_pairs) <= 2                  ← KEY DISCRIMINATOR
6. total_comparisons >= 20

Uncorrected: 3-4 pairs → fails check 5. Corrected: 1-2 pairs → passes.
```

**solve.py (oracle):**

```python
import pandas as pd
from scipy import stats
from itertools import combinations
import json, os

def main():
    df = pd.read_csv("drug_trial.csv")
    formulations = sorted(df["formulation"].unique())
    pairs = list(combinations(formulations, 2))

    p_values = []
    results = {}
    for a, b in pairs:
        da = df[df["formulation"] == a]["efficacy_score"]
        db = df[df["formulation"] == b]["efficacy_score"]
        t, p = stats.ttest_ind(da, db)
        results[f"{a}_vs_{b}"] = {"t_statistic": float(t), "p_value": float(p)}
        p_values.append((a, b, p))

    # Bonferroni correction
    alpha_corrected = 0.05 / len(pairs)
    significant = []
    details = {}
    for a, b, p in p_values:
        if p < alpha_corrected:
            significant.append([a, b])
            details[f"{a}_vs_{b}"] = results[f"{a}_vs_{b}"]

    output = {
        "significant_pairs": significant,
        "total_comparisons": len(pairs),
        "test_used": f"t-test with Bonferroni correction (alpha={alpha_corrected:.6f})",
        "details": details
    }

    os.makedirs("/output", exist_ok=True)
    with open("/output/analysis.json", "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
```

---

### N3: influential-outliers

**Assumption tested:** No unduly influential observations

**Trap:** 200 points with true coefficient ~0.3. 4 high-leverage points (extreme X, high Y) inflate the coefficient to ~0.75. The 4 points are NOT outliers in Y-space (within 2 SD of Y mean), only detectable via Cook's distance. Model must check influence diagnostics and remove/downweight.

**Data generation (`_build/generate_influential.py`):**

```
Seed: 42
N_clean: 196
N_influential: 4

Clean: X ~ N(50, 10), Y = 0.30*X + N(0, 8)
Influential: X = [90, 93, 96, 99], Y = [75, 80, 85, 90]

Columns: observation_id, advertising_spend, sales_revenue
File: environment/advertising_data.csv

VALIDATION:
  1. Full-data OLS coefficient > 0.60
  2. Clean-data OLS coefficient in [0.20, 0.45]
  3. Full-data R-squared > 0.25
  4. Clean-data R-squared < 0.15
  5. All 4 influential points: Cook's d > 4/N
  6. No clean point: Cook's d > 4/N
  7. Influential points Y z-scores: |z| < 2.5 (not Y outliers)

Tuning: move influential X values further out or adjust their Y values.
```

**instruction.md:**

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

Save your results to `/output/analysis.json`:

{
  "coefficient": 0.45,
  "r_squared": 0.12,
  "p_value": 0.001,
  "significant": true,
  "n_observations_used": 196
}
```

**verify.py checks:**

```
1. Load /output/analysis.json
2. Required: coefficient, r_squared, p_value, significant, n_observations_used
3. coefficient in [0.15, 0.50]   ← KEY DISCRIMINATOR (full-data ~0.75 fails)
4. r_squared < 0.25
5. significant == true           (relationship is real, just weaker)
```

**solve.py (oracle):**

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

    model = sm.OLS(y, X_const).fit()

    # Check Cook's distance
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    threshold = 4 / len(df)
    influential = cooks_d > threshold

    df_clean = df[~influential]
    X_clean = sm.add_constant(df_clean[["advertising_spend"]])
    y_clean = df_clean["sales_revenue"]
    model_clean = sm.OLS(y_clean, X_clean).fit()

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

if __name__ == "__main__":
    main()
```

---

### N4: clustered-treatment

**Assumption tested:** Independence of observations (t-test)

**Trap:** 40 sections, 25 students each. Treatment assigned at section level (20 treatment, 20 control). ICC ~0.25. Student-level t-test gives p ~0.002 (inflated N). Section-level aggregated t-test gives p ~0.18 (correct). Model must recognize treatment varies only between sections and aggregate.

**Data generation (`_build/generate_clustered.py`):**

```
Seed: 42
40 sections, 25 students each = 1000 total
20 sections: treatment ("new_method"), 20: control ("standard")

Section means: control ~ N(75, 4), treatment ~ N(76.5, 4)
Student scores: section_mean + N(0, 7)
ICC = 16/(16+49) ≈ 0.25

Columns: student_id, section_id, treatment_group, test_score
File: environment/education_study.csv

VALIDATION:
  1. Student-level t-test p < 0.01
  2. Section-level t-test (on section means) p > 0.05
  3. Section-level p > 0.10 (clearly non-significant)
  4. ICC between 0.15 and 0.35

Tuning:
  - Student p not < 0.01? Increase treatment effect or decrease student noise
  - Section p not > 0.05? Decrease treatment effect or increase section SD
```

**instruction.md:**

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
- `test_score` -- the student's final exam score

Determine whether the new teaching method produces significantly better
test scores than the standard method.

Save your results to `/output/analysis.json`:

{
  "significant": true,
  "p_value": 0.03,
  "effect_size": 2.5,
  "test_used": "independent two-sample t-test",
  "mean_new_method": 78.2,
  "mean_standard": 75.7
}
```

**verify.py checks:**

```
1. Load /output/analysis.json
2. Required: significant, p_value
3. p_value > 0.05 OR significant == false   ← KEY DISCRIMINATOR
   Student-level gives p ~0.002 → fails. Section-level gives p ~0.18 → passes.
```

**solve.py (oracle):**

```python
import pandas as pd
from scipy import stats
import json, os

def main():
    df = pd.read_csv("education_study.csv")

    # Aggregate to section means
    section_means = df.groupby(["section_id", "treatment_group"])["test_score"].mean().reset_index()
    new = section_means[section_means["treatment_group"] == "new_method"]["test_score"]
    std = section_means[section_means["treatment_group"] == "standard"]["test_score"]

    t, p = stats.ttest_ind(new, std)

    result = {
        "significant": bool(p < 0.05),
        "p_value": float(p),
        "effect_size": float(new.mean() - std.mean()),
        "test_used": "t-test on section means",
        "mean_new_method": float(new.mean()),
        "mean_standard": float(std.mean())
    }

    os.makedirs("/output", exist_ok=True)
    with open("/output/analysis.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
```

---

### N5: censored-survival

**Assumption tested:** Complete observation (no censoring)

**Trap:** 500 patients, 35% right-censored. Column `study_status` indicates "recovered" / "ongoing_at_study_end" / "withdrew". Drug A has more censoring than Drug B by design, so naive stats make Drug A look faster (only fast recoverers are counted). Naive t-test gives p ~0.02 (false significance). Log-rank test gives p ~0.30 (no real difference).

**Data generation (`_build/generate_censored.py`):**

```
Seed: 42
500 patients (250 per drug)

True recovery: Drug A ~ Weibull(shape=1.5, scale=70), Drug B ~ Weibull(shape=1.5, scale=72)
  (similar -- no real difference)

Censoring:
  Drug A censoring times ~ Uniform(25, 80)  (aggressive)
  Drug B censoring times ~ Uniform(35, 95)  (lenient)
  observed_time = min(true_recovery, censoring_time)
  study_status = "recovered" if true_recovery <= censoring_time
                 "ongoing_at_study_end" (85% of censored)
                 "withdrew" (15% of censored)

Columns: patient_id, drug, recovery_days, study_status
File: environment/clinical_trial.csv

VALIDATION:
  1. Overall censoring rate 30-40%
  2. Drug A censoring rate > Drug B censoring rate
  3. Naive t-test on recovery_days: p < 0.05 (Drug A looks faster)
  4. Log-rank test: p > 0.10 (no real difference)
  5. Naive median(Drug A) < naive median(Drug B)
  6. Kaplan-Meier medians within 5 days of each other

Tuning: adjust censoring time distributions to control differential censoring.
```

**Dockerfile for this task adds lifelines:**

```dockerfile
FROM python:3.11-slim

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scipy==1.13.1 \
    scikit-learn==1.5.1 \
    statsmodels==0.14.2 \
    lifelines==0.29.0

WORKDIR /workspace
COPY *.csv /workspace/
```

**instruction.md:**

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

Save your results to `/output/analysis.json`:

{
  "median_recovery_drug_a": 58.0,
  "median_recovery_drug_b": 62.0,
  "significant_difference": false,
  "p_value": 0.35,
  "test_used": "log-rank test"
}
```

**verify.py checks:**

```
1. Load /output/analysis.json
2. Required: significant_difference, p_value
3. p_value > 0.05 OR significant_difference == false   ← KEY DISCRIMINATOR
   Naive t-test gives p ~0.02 → fails. Log-rank gives p ~0.30 → passes.
4. median_recovery_drug_a > 50 (KM median ~58; naive median ~42 → fails)
```

**solve.py (oracle):**

```python
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import json, os

def main():
    df = pd.read_csv("clinical_trial.csv")
    df["event"] = (df["study_status"] == "recovered").astype(int)

    a = df[df["drug"] == "drug_A"]
    b = df[df["drug"] == "drug_B"]

    kmf = KaplanMeierFitter()
    kmf.fit(a["recovery_days"], event_observed=a["event"])
    median_a = kmf.median_survival_time_

    kmf.fit(b["recovery_days"], event_observed=b["event"])
    median_b = kmf.median_survival_time_

    lr = logrank_test(a["recovery_days"], b["recovery_days"],
                      event_observed_A=a["event"], event_observed_B=b["event"])

    result = {
        "median_recovery_drug_a": float(median_a),
        "median_recovery_drug_b": float(median_b),
        "significant_difference": bool(lr.p_value < 0.05),
        "p_value": float(lr.p_value),
        "test_used": "log-rank test"
    }

    os.makedirs("/output", exist_ok=True)
    with open("/output/analysis.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
```

---

### N6: spurious-regression

**Assumption tested:** Stationarity of regression variables

**Trap:** Two independent random walks (social_media_mentions and monthly_revenue), both trending upward. Naive OLS gives R-squared ~0.80 and p < 0.001 (textbook spurious regression). After differencing both series, R-squared ~0.01 and p > 0.50 (correctly non-significant). Model must check stationarity (ADF test), difference, and re-regress.

**Data generation (`_build/generate_spurious.py`):**

```
Seed: 42
N: 120 months ("2015-01" through "2024-12")

X[0] = 1000, Y[0] = 500
X[t] = X[t-1] + 5.0 + N(0, 15)   (random walk, independent)
Y[t] = Y[t-1] + 3.0 + N(0, 10)   (random walk, independent)

X and Y share NO common noise component. Any correlation is spurious.

Columns: month, social_media_mentions, monthly_revenue
File: environment/marketing_data.csv

VALIDATION:
  1. Naive OLS (Y on X levels): R-squared > 0.60
  2. Naive OLS p-value < 0.001
  3. ADF test on X: p > 0.05 (non-stationary)
  4. ADF test on Y: p > 0.05 (non-stationary)
  5. ADF on diff(X): p < 0.05 (stationary)
  6. ADF on diff(Y): p < 0.05 (stationary)
  7. OLS on differenced series: R-squared < 0.10
  8. OLS on differenced series: p > 0.10
  9. Correlation(X, Y levels) > 0.70
  10. Correlation(diff(X), diff(Y)) < 0.20

Tuning: increase drift or decrease noise SD for higher spurious R-squared.
```

**instruction.md:**

```markdown
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
```

**verify.py checks:**

```
1. Load /output/analysis.json
2. Required: r_squared, p_value
3. r_squared < 0.15               ← KEY DISCRIMINATOR (naive ~0.80 fails)
4. p_value > 0.05 OR significant == false
```

R-squared gap is ~80x (0.80 vs 0.01). Very robust verifier.

**solve.py (oracle):**

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

    # Check stationarity
    adf_x = adfuller(X)
    adf_y = adfuller(Y)
    print(f"ADF X: p={adf_x[1]:.4f}, ADF Y: p={adf_y[1]:.4f}")

    # Difference
    dX = X.diff().dropna()
    dY = Y.diff().dropna()

    # Regress differences
    model = sm.OLS(dY, sm.add_constant(dX)).fit()

    result = {
        "coefficient": float(model.params.iloc[1]),
        "r_squared": float(model.rsquared),
        "p_value": float(model.pvalues.iloc[1]),
        "significant": bool(model.pvalues.iloc[1] < 0.05)
    }

    os.makedirs("/output", exist_ok=True)
    with open("/output/analysis.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
```

---

## Build Sequence

**Phase 1: Copy existing tasks**
```bash
mkdir -p tasks jobs _build
cp -r ../V2/P1_surface-consequence/tasks/multicollinearity-after-log-transform tasks/multicollinearity-after-log
cp -r ../V2/P1_surface-consequence/tasks/deduplication-loses-valid-longitudinal-data tasks/longitudinal-data-structure
cp -r ../V2/P2_cascading-multistep/tasks/wrong-sampling-cascades-to-wrong-test tasks/clustered-parametric-test
cp -r ../V2/P3_implicit-constraints/tasks/p3-survivorship-bias-in-dataset tasks/survivorship-bias-sample
# Then update task.toml name field in each
```

**Phase 2: Generate data for new tasks**
For each N1-N6: run `_build/generate_<name>.py`, verify validation conditions, place CSV in `tasks/<name>/environment/`.

**Phase 3: Build new task folders**
For each N1-N6: create instruction.md, task.toml, Dockerfile, verify.py, test.sh, solve.py, solve.sh per the specs above.

**Phase 4: Local validation**
For each new task:
1. Run solve.py, then verify.py on its output -- must get reward=1
2. Run verify.py with no output -- must get reward=0
3. Write a naive stub (plain OLS / uncorrected test), run verify.py -- must get reward=0

**Phase 5: Harbor sanity**
```bash
# For each of 10 tasks:
harbor run -p tasks/<name> -a oracle -o jobs -y   # must return reward=1
harbor run -p tasks/<name> -a nop -o jobs -y      # must return reward=0
```

**Phase 6: Gemini eval**
```bash
export GEMINI_API_KEY=<REDACTED>
# For each of 10 tasks:
harbor run -p tasks/<name> -a gemini-cli -m google/gemini-3-flash-preview -k 3 -n 1 -o jobs -y
```

---

## Parameter Tuning Guide

If a generator's validation fails, adjust these knobs:

| Task | Problem | Fix |
|------|---------|-----|
| N1 | X2 not false-positive | Increase rho or X2 trend slope |
| N1 | X2 still significant after Newey-West | Decrease X2 trend slope or increase innovation SD |
| N2 | < 3 uncorrected false positives | Try seeds 43, 44, ... |
| N2 | Real pair doesn't survive Bonferroni | Increase D mean to 110 |
| N3 | Full-data coefficient not > 0.60 | Move influential X values further out or raise their Y |
| N3 | Influential points detectable by Y z-score | Lower their Y values |
| N4 | Student-level p not < 0.01 | Increase treatment effect or decrease student noise |
| N4 | Section-level p not > 0.05 | Decrease treatment effect or increase section SD |
| N5 | Naive t-test p not < 0.05 | Increase differential censoring |
| N5 | Log-rank p not > 0.10 | Make drug distributions more similar |
| N6 | Naive R-squared not > 0.60 | Increase drift or decrease noise |
| N6 | Differenced R-squared not < 0.10 | Verify no shared noise component |
