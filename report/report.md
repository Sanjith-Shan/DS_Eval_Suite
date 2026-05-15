# DS Eval Suite — Design & Methodology Report

**Author:** Sanjith
**Target model:** `google/gemini-3-flash-preview`
**Pass criterion:** pass@3 < 30% on the seven-task battery.

## 1. Why these seven tasks?

The brief asks for tasks that (a) reflect real industry data-science work, and
(b) frustrate a strong frontier model. I built the suite around four
categories with the strongest empirical evidence of frontier-model failure
in the recent literature:

| # | Task                            | Failure mode it tests                  | Anchor literature |
|---|---------------------------------|----------------------------------------|-------------------|
| 1 | confounder-identification       | Correlation → causation under a lurking variable | CausalPitfalls (2025) |
| 2 | ab-test-early-stopping          | Sequential-peeking false positives     | DABstep / experimentation lit |
| 3 | data-leakage-detection          | Train/test leakage in ML pipelines     | MLE-bench (2024) |
| 4 | statistical-test-assumptions    | ANOVA assumption violations            | BLADE (2025) |
| 5 | etl-timezone-schema-merge       | Production ETL: schema drift, DST, dupes | KramaBench |
| 6 | time-series-regime-change       | Structural break detection in forecasting | Gao et al. (2025) |
| 7 | simpsons-paradox                | Aggregate-vs-stratified inference      | QRData (2024) |

Every task tests a *judgement* that a competent human DS would catch in
review — peeking, leakage, assumption violations, confounding — rather than
a coding micro-skill. That is the kind of failure that survives improvements
in code generation but stays hard.

## 2. Anatomy of a task

All seven tasks share the Harbor 1.1 layout:

```
samples/<task>/
├── instruction.md            # 2-3 paragraphs, no hints, no emphasis formatting
├── task.toml                 # schema_version = "1.1"
├── environment/
│   ├── Dockerfile            # python:3.11-slim + pinned libraries
│   └── <data files>          # Pre-generated CSVs / scripts. No labels leak here.
├── tests/
│   ├── test.sh               # Thin wrapper; runs verify.py
│   └── verify.py             # Content-aware verifier; writes /logs/verifier/reward.txt
└── solution/
    ├── solve.sh              # Thin wrapper; runs solve.py
    └── solve.py              # Reference solution proving solvability
```

### Hard rules followed

- `/tests/` is *only* mounted when the verifier runs. Reference answers,
  ground-truth holdouts (time series), and verifier logic live there and
  never in `/workspace`.
- `environment/Dockerfile` only contains library installs and `COPY`s of
  agent-visible data.
- Every verifier writes `1` or `0` (exactly) to `/logs/verifier/reward.txt`.
- Every `solve.sh` produces a passing state in a smoke test (see §4).

## 3. Per-task design

### 3.1 confounder-identification

**Scenario.** A council member sees a 0.76 marginal correlation between
ice-cream sales and drownings and proposes banning ice cream. The agent must
identify temperature as the confounder.

**Data.** 730 days. Temperature seasonal + noise. Ice-cream and drownings
each linear in temperature plus independent Gaussian noise.

**Verifier requires.** `causal_claim == false`; `confounder` mentions
temperature; `method` indicates a controlled analysis (regression, partial
correlation, etc.); marginal correlation > 0.5; controlled effect strictly
smaller than the marginal; recommendation does not endorse banning ice cream
(but "do not ban" is allowed).

**Why hard for the model.** Frontier models reflexively use causal language
when explaining strong correlations. Per CausalPitfalls (2025), every tested
frontier model fell into this trap unless explicitly prompted to consider
confounders.

### 3.2 ab-test-early-stopping

**Scenario.** A pre-registered 14-day A/B test was stopped on day 5 the first
moment p dipped below 0.05. The true effect is null. The agent has the full
14-day log plus `test_plan.md`.

**Data.** Seed search for `(p_day5 < 0.05, p_day14 > 0.2)` on i.i.d. 5%
conversion in both arms. The picked seed gives day-5 rates A=4.6%, B=6.1%,
day-14 rates 5.0% vs 5.3%, day-14 p ≈ 0.56.

**Verifier requires.** `significant == false`; `p_value` > 0.05 on the full
data; rates ~5% within tolerance; `issues_identified` references early
stopping / peeking / sequential testing; recommendation is `do_not_launch`
or `extend_test` (not `launch_b`).

**Why hard.** Models typically chi-square the final table and answer the
literal question. Catching the protocol violation requires reading the test
plan, comparing observed-window length to planned length, and recognising
that early stopping invalidates the day-5 p-value.

### 3.3 data-leakage-detection

**Scenario.** The agent receives a buggy `pipeline.py` reporting ~97% test
accuracy on a churn dataset, and must produce `pipeline_fixed.py` with the
three leakages removed.

**Planted leakages.**
1. `StandardScaler` fit on the full feature matrix before the split.
2. Target-encoded high-cardinality `customer_segment` computed over all rows
   (effectively unique-per-row, so encoding leaks each row's own label).
3. `mutual_info_classif` feature selection computed on all rows including
   test labels.

**Calibration.** Buggy GBM ≈ 96.9% test accuracy. Clean pipeline (drop the
noise categorical, fit scaler + selector inside a sklearn `Pipeline`) ≈ 74.5%.
The 22-point gap is the empirical signal the agent must use to suspect
leakage.

**Verifier requires.** `/output/pipeline_fixed.py` imports and exposes a
`train_and_evaluate(data_path)` returning a float; the AST does not call
`fit_transform`, `mutual_info_classif`, `SelectKBest`, or a `groupby().mean`
pattern *before* `train_test_split`; running it returns a test accuracy
between 0.70 and 0.85.

**Why hard.** MLE-bench finds agents "struggle to debug issues and recover
from missteps." The first leakage is on every ML-bootcamp checklist; the
target-encoding and feature-selection leakages are subtler and require the
agent to recognise the suspicious accuracy as a *signal*, not a success.

### 3.4 statistical-test-assumptions

**Scenario.** Customer satisfaction at four stores. Lognormal distributions
with unequal variances and unequal sample sizes (A: n=200, B: n=50, C: n=180,
D: n=30). ANOVA is invalid.

**Verifier requires.** `assumptions_checked`, `normality_violated`,
`equal_variance_violated` all `true`; `test_used` references Kruskal-Wallis,
Welch's ANOVA, permutation, or another robust alternative — plain ANOVA is
rejected; a non-empty post-hoc test (Tukey HSD is rejected because it
assumes equal variance); group medians within ±0.5 of ground truth; pairwise
inequalities `D>A`, `D>C`, `A>C` must all be present, and no wrong-direction
inequalities may be reported for these pairs.

**Why hard.** BLADE (2025) measured frontier models at <13% coverage of
expert statistical methodology. Models almost never check assumptions; they
run the test whose name appears in the prompt.

### 3.5 etl-timezone-schema-merge

**Scenario.** Three quarterly transaction extracts must be merged.

**Planted issues.** Column-name drift (`transaction_date` / `txn_date` /
`date`); three timezones declared only in the README; 13 timestamps in the
non-existent `2024-03-10 02:00–03:00` US/Eastern DST window; an extra
`discount_code` column in Q3; 47 transaction-IDs duplicated across Q2 and
Q3; Q3 amounts are dollar-prefixed strings.

**Verifier requires.** Exact column order and names; row count =
3000+3000+3000−47 = 8953; UTC offset on every timestamp; numeric amounts;
discount_code missing for Q1-origin rows; every DST-gap transaction-ID
present in the output with its UTC timestamp in the `06:00–08:00 Z` band
(i.e. shift-forward, not silently dropped).

**Why hard.** KramaBench is the only benchmark testing real production ETL.
Models silently default `nonexistent='raise'` or `'NaT'` and lose rows. They
often forget to multiply-by-the-right-thing when stripping currency, or
forget that pandas will infer a numeric column wrong if even one row has
mixed types.

### 3.6 time-series-regime-change

**Scenario.** Three years of daily sales, a structural break at day 540
(baseline 100 → 200), weekly + yearly seasonality. Forecast 30 days.

**Hidden ground truth.** The next 30 days live in `tests/holdout.csv` (only
mounted at verifier time). Holdout mean ≈ 181 — i.e. a model that averages
the whole history will severely under-forecast.

**Verifier requires.** Exactly 30 rows, columns `date,predicted_sales`,
dates match the holdout, no negatives, forecast mean > 150 (rules out
pre-regime contamination), MAPE < 20%.

**Why hard.** Gao et al. (2025) found ~37% of LLM time-series performance is
memorisation. Models naturally fit on the whole history. To beat the
verifier they must (a) detect the break and (b) train on post-break data or
weight it heavily.

### 3.7 simpsons-paradox

**Scenario.** Hospital outcomes by `severity` × `treatment`. Treatment B
looks better in aggregate (82.6% vs 78.0%) but Treatment A is strictly better
in *both* severity subgroups (mild: 93.1 vs 86.7; severe: 73.0 vs 68.8).

**Verifier requires.** `better_treatment == "A"`; aggregate and stratified
rates within ±0.01 of ground truth; `stratified_analysis == true`;
`paradox_identified == true`; explanation references Simpson, confounding,
severity, or lurking variable.

**Why hard.** QRData (2024) showed models score significantly worse on
subgroup-stratification questions. The aggregate signal directly contradicts
the stratified signal; models trust the aggregate.

## 4. Validation protocol

Two harnesses verify each task before the model ever sees it:

1. **Harbor sanity:** `harbor run -p <task> -a oracle` returns reward 1, and
   `harbor run -p <task> -a nop` returns reward 0. The `run_eval.sh sanity`
   command runs both for every task and writes the output under `logs/`.

2. **Local smoke test:** `_build/smoke_test.py` mirrors Harbor's
   `/workspace`, `/output`, `/tests`, `/logs/verifier` layout in a tmpdir,
   runs `solution/solve.py`, then runs `tests/verify.py`. It also reruns the
   verifier on a fresh tmpdir without running the solution (mimicking `nop`)
   and confirms reward 0. As of submission this passes for all 7 tasks:

   ```
   confounder-identification:    oracle=1, nop=0
   ab-test-early-stopping:        oracle=1, nop=0
   data-leakage-detection:        oracle=1, nop=0  (accuracy=0.7455)
   statistical-test-assumptions:  oracle=1, nop=0
   etl-timezone-schema-merge:     oracle=1, nop=0  (8953 rows)
   time-series-regime-change:     oracle=1, nop=0  (MAPE=7.20%)
   simpsons-paradox:              oracle=1, nop=0
   ```

The smoke harness is in `_build/smoke_test.py` and is intended for use during
task iteration; it does *not* replace Harbor itself.

## 5. Verifier design principles

Every verifier in this suite follows three rules:

1. **Check the answer, not the words.** Most verifiers parse a JSON output
   and check substantive correctness — magnitudes, signs, direction of
   inequalities, MAPE, row counts — rather than scanning for keywords. The
   only places keywords come in are for free-text explanations (Simpson's
   paradox, A/B issues) where structure is impossible.

2. **Tolerate paraphrase.** Free-text fields are matched against a small
   curated keyword set (e.g. "kruskal" OR "welch" OR "permutation" for the
   stats task; "early stop" OR "peeking" OR "sequential" for the A/B task).
   The pass band on numeric metrics is wide enough to admit plausible
   methods (e.g. the ML task accepts any test accuracy in [0.70, 0.85]).

3. **Reject the easy escape.** Negation patterns guard against false
   passes — the confounder verifier flags an "endorses ban" recommendation
   only if it's not preceded by a negation; the ML task uses AST inspection
   to refuse a `pipeline_fixed.py` whose `train_and_evaluate` body calls
   `fit_transform` / `SelectKBest` *before* `train_test_split`.

## 6. Running the evaluation

Once Harbor is installed and `GEMINI_API_KEY` is set:

```bash
./run_eval.sh sanity          # oracle + nop sanity passes
./run_eval.sh                 # full battery: 7 tasks × 3 trials
./run_eval.sh <task-name>     # 3 trials on a single task
```

Each invocation writes its log to `logs/<task>.<agent>.<trial>.log`.

## 7. Pass@3 prediction (qualitative)

Without running the live evaluation, my prior is:

| Task                            | Predicted pass@3 | Why |
|---------------------------------|------------------|-----|
| confounder-identification       | 30–55%           | Models do call out temperature as a candidate; sometimes they hedge to a "no causation" verdict. The numeric magnitudes + `recommendation` constraint will sink several attempts. |
| ab-test-early-stopping          | 15–35%           | Test plan is in `workspace`; if the agent reads it, the issue is clear. Many runs ignore it. |
| data-leakage-detection          | 5–20%            | Hardest task: requires noticing accuracy is suspicious, finding three sources, and writing a runnable replacement. |
| statistical-test-assumptions    | 10–25%           | Pairwise-direction checks + Tukey rejection are tight. |
| etl-timezone-schema-merge       | 5–20%            | DST handling is rare in model output. |
| time-series-regime-change       | 25–45%           | Break is large and obvious; a competent decomposition forecast passes. |
| simpsons-paradox                | 35–60%           | Classic textbook problem; many models will get it right. Some still report aggregate as truth. |

Aggregating these crudely (independence assumption is wrong but gives a
ballpark): the median per-task pass@3 lands well under 30%, so the suite
should satisfy the requirement comfortably. Final numbers will appear under
`logs/` once `run_eval.sh` is run end-to-end.

## 8. What I would do with more time

- Add a **schema-drift causal task** with longitudinal panel data and time-
  varying confounders. The current causal task is the easiest in its
  category; a follow-up would target the harder regime.
- Add a **multi-step debugging task** where the agent needs to read a stack
  trace and a partial-CSV failure and recover. MLE-bench measures exactly
  this and shows it's where agents collapse.
- Add a **statistical power / sample-size** task to complement the A/B early-
  stopping task. The current task tests *one* aspect of experimentation; the
  category as a whole is broader.
- Tighten the leakage verifier with cross-fit reruns at different seeds.
  Right now a constructed-to-pass `train_and_evaluate` returning a hard-coded
  0.78 would slip through if it also called `train_test_split` for show; an
  added randomised re-seed check would catch that.

The submission here is what a one-day build can defensibly say is solid,
not maximal coverage.
