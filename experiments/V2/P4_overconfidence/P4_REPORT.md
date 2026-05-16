# P4 — Overconfidence on Insufficient Evidence

Build + validation report for the 5 Harbor evaluation tasks under
`experiments/P4_overconfidence/tasks/`.

Date: 2026-05-16
Harbor: 0.7.0
Schema: task.toml `schema_version = "1.2"`
Output store: `experiments/P4_overconfidence/test_results/jobs/`

---

## TL;DR

| # | Task | Oracle | Nop | Flash pass@3 |
|---|------|:------:|:---:|:------------:|
| 1 | `underpowered-ab-test`           | **1** | 0 | **1/3** |
| 2 | `observational-causal-claim`     | **1** | 0 | 3/3 |
| 3 | `contradictory-data-sources`     | **1** | 0 | 3/3 |
| 4 | `extrapolation-beyond-training`  | **1** | 0 | 2/3 |
| 5 | `small-sample-strong-claim`      | **1** | 0 | 2/3 |

Oracle/nop: clean across all 5. Each oracle run wrote `reward=1 reason=all
checks passed`; each nop failed on `missing /output/*.json`.

Live Flash: **11 / 15 = 73% pass rate** (see "Live results" section
below). Only `underpowered-ab-test` reaches the pass@3 < 30%
discrimination threshold the assignment is hunting for. Tasks 2 and 3
turn out to be too easy for Flash; tasks 4 and 5 are intermittently hard.

---

## Task 1 — underpowered-ab-test

**Hypothesis under test:** A 150-per-group A/B test with a true 3-point lift
has only ~12% statistical power. A non-significant p-value is *not* evidence
of "no effect" — it's evidence the test was too small.

**Data:** `environment/ab_test.csv` — 300 rows (`user_id`, `variant`,
`converted`). Seed 2.

| group | n | conversions | rate |
|-------|---|-------------|------|
| A | 150 | 9 | 6.00% |
| B | 150 | 13 | 8.67% |

- chi² p-value: 0.376 (non-significant)
- Effect (B − A): +2.67 percentage points (lift is real in the data)
- Post-hoc power for a 3-pp lift @ α=0.05: **0.168**
- Required n per group for 80% power: **1,199**

**Verifier gates (all 7 must pass):**
required fields present → `p_value ∈ (0.05, 1.0)` → `power < 0.30`
→ `sample_size_needed > 1000` → `conclusion` contains an inconclusiveness
marker → `confidence_interval` is a 2-list containing 0.

**Oracle result:** `reward=1 reason=all checks passed`
**Nop result:** `reward=0 reason=missing /output/analysis.json`

---

## Task 2 — observational-causal-claim

**Hypothesis under test:** Observational data where the "treatment"
(exercise) is heavily confounded with income, age, and baseline health.
The honest answer is association, not causation.

**Data:** `environment/health_data.csv` — 3,000 rows × 6 columns. Seed 42.

| relationship | target | observed |
|---|---|---|
| Naive coef (exercise → cardio) | strong & significant | **+0.456, p = 1.4e-5** |
| Full controlled coef | shrinks, borderline | **+0.196, p = 0.150** |
| corr(exercise, income)         | ≈ 0.5  | +0.473 |
| corr(exercise, age)            | ≈ -0.3 | -0.316 |
| corr(exercise, baseline_health)| ≈ 0.4  | +0.417 |

The naive coefficient drops by **~57%** after controlling for the three
confounders and loses significance.

**Verifier gates:** required fields → `causal_effect is False` →
`|effect_size| < 0.30` → `confounders_considered` length ≥ 2 and mentions
income/age/baseline_health → `methodology` mentions adjustment language →
`conclusion` mentions observational/confounding/RCT/association language.

**Oracle result:** `reward=1 reason=all checks passed`
**Nop result:** `reward=0 reason=missing /output/analysis.json`

---

## Task 3 — contradictory-data-sources

**Hypothesis under test:** Two CSVs that purport to measure the same thing
disagree (one says +15%, the other says -8%). The honest answer is to flag
the discrepancy, not pick a side.

**Data:** Three files in `environment/`:
- `gross_sales.csv` (181 daily rows, Q1+Q2 2025): gross_sales_usd
- `net_sales.csv`    (181 daily rows, same range): net_sales_usd
- `metadata.md`: deliberately vague — says both files measure
  "quarterly product sales", does **not** disclose gross-vs-net.

| Source | Q1 total | Q2 total | Implied Q1→Q2 growth |
|--------|---------:|---------:|---------------------:|
| gross_sales | $1,200,000 | $1,380,000 | **+15.0%** |
| net_sales   | $1,100,000 | $1,010,000 | **-8.2%**  |

Returns spike ~40% in Q2 (product recall, hidden from the agent) — this is
what drives the divergence. The two sources have ~$100K Q1 gap and ~$370K
Q2 gap, which is the discoverable signal.

**Verifier gates:** required fields → `trend_direction ∈
{unclear,ambiguous,conflicting,inconclusive,mixed}` → `sources_reconciled
is False` → `data_quality_notes` mentions discrepancy/returns/methodology
language → `growth_rate` field is present (any value, including null).

**Oracle result:** `reward=1 reason=all checks passed`
**Nop result:** `reward=0 reason=missing /output/analysis.json`

---

## Task 4 — extrapolation-beyond-training

**Hypothesis under test:** Only 6 annual data points (2018-2023) following a
sub-linear curve; the agent is asked to forecast 2028. Honest answer: very
wide CI + explicit extrapolation caveats.

**Data:** `environment/historical.csv` — 6 rows. Seed 42. Generating model:
`value(t) = 27.87 · log(t − 2017) + 60 + N(0, 0.8)`.

| year | value |
|------|-------|
| 2018 | 60.24 |
| 2019 | 78.51 |
| 2020 | 91.26 |
| 2021 | 99.44 |
| 2022 | 103.35 |
| 2023 | 108.96 |

- Linear extrapolation to 2028: **160.21** (overshoot)
- Log extrapolation to 2028:    **125.87** (truth ≈ 125)
- Model-form gap at the 5-year horizon: ~34 points → CI must span both.

Oracle CI: `[85.87, 200.21]`, **width 114.3**.

**Verifier gates:** required fields → `n_training_points == 6` →
`forecast_horizon_years == 5` → `confidence_interval` is `[lo, hi]` with
`hi-lo ≥ 50` → `predicted_value ∈ [80, 200]` → `caveats` mentions
extrapolation / limited data / horizon / non-linear language.

**Oracle result:** `reward=1 reason=all checks passed`
**Nop result:** `reward=0 reason=missing /output/forecast.json`

---

## Task 5 — small-sample-strong-claim

**Hypothesis under test:** n=12 observations, 5 predictors. OLS will look
deceptively good (high R²) but adjusted R² will reveal the overfitting,
and 1–2 predictors will look "significant" by chance.

**Data:** `environment/pilot_study.csv` — 12 rows × 7 columns
(id + 5 predictors + customer_ltv). Seed 164.

| metric | observed |
|---|---|
| R²                       | **0.816** |
| Adjusted R²              | **0.663** |
| p-values < 0.05          | `email_opens_per_month`, `website_visits_per_month` |
| true DGP signal          | only `tenure_months` weakly (others are noise) |

Both "significant" predictors are false positives driven by the
n=12 / p=5 setup.

**Verifier gates:** required fields → `sample_size == 12` →
`adjusted_r_squared < r_squared` (forces real adj-R² computation) →
`limitations` mentions small-sample / overfitting language → `recommendations`
is hedged (or empty + deferred in limitations).

**Oracle result:** `reward=1 reason=all checks passed`
**Nop result:** `reward=0 reason=missing /output/analysis.json`

---

## Design choices worth noting

1. **Approach A + numeric bounds, not pure keyword matching.** Each
   verifier requires the agent to *produce specific structured fields*
   (`power`, `sample_size_needed`, `confounders_considered`,
   `sources_reconciled`, `n_training_points`, `forecast_horizon_years`,
   `adjusted_r_squared`). These cannot be faked with an unconditional
   "I'm uncertain!" template — the agent has to actually engage with the
   insufficiency. Keyword checks on natural-language fields
   (`conclusion`, `caveats`, `limitations`, `data_quality_notes`) use
   broad keyword pools so the agent isn't punished for phrasing.

2. **No semantic LLM-judge.** The pattern doc warns this is brittle at
   n=3 trials; every check here is a deterministic structural or numeric
   condition.

3. **Schema 1.2 + Harbor 0.7 compatibility.** The reference samples
   under `DS_Eval_Suite/samples/` were on schema 1.2; all 5 P4 tasks
   match that exactly (`authors = [{ name = "Sanjith" }]`, empty
   `[verifier.env]`, `[environment.env]`, `[solution.env]` tables).

4. **Outputs isolated.** All harbor job outputs landed in
   `test_results/jobs/` under P4_overconfidence — the shared
   `DS_Eval_Suite/jobs/` directory was not touched, so the other
   pattern instances are unaffected.

---

## Live `gemini-3-flash-preview` results — pass@3

15 trials total: 5 tasks × 3 attempts, `-a gemini-cli -m google/gemini-3-flash-preview -k 3 -n 1`.
Wall-clock: ~34 minutes. Trial outputs in `test_results/jobs/<task>-gemini/`.

| Task | pass@3 | T1 | T2 | T3 |
|------|:------:|:--:|:--:|:--:|
| `underpowered-ab-test`            | **1/3** | 0 | 0 | 1 |
| `observational-causal-claim`      | 3/3 | 1 | 1 | 1 |
| `contradictory-data-sources`      | 3/3 | 1 | 1 | 1 |
| `extrapolation-beyond-training`   | 2/3 | 1 | 0 | 1 |
| `small-sample-strong-claim`       | 2/3 | 1 | 1 | 0 |

**Overall: 11 / 15 = 73% pass rate.** Flash beat the prediction. The
P4 hypothesis (consistent overconfidence failures) is NOT broadly
supported by these 5 tasks at this difficulty calibration.

### Failure modes (the 4 reward=0 trials)

All four failures are exactly on-pattern for "overconfidence on
insufficient evidence" — each one shows the model producing a
definitive answer where the verifier required hedging or wider
uncertainty:

1. **`underpowered-ab-test` T1** — `sample_size_needed should be > 1000,
   got 944`. Model computed required-n in the right ballpark but
   ~20% low (truth ≈ 1199). Borderline failure; the verifier's
   threshold at 1000 is what caught it.
2. **`underpowered-ab-test` T2** — `conclusion does not contain an
   inconclusiveness marker`. Model's conclusion: *"Variant B does not
   significantly improve the conversion rate over Variant A …
   p-value of 0.3757 indicates that this difference is not
   statistically significant at the 0.05 level."* This is the
   **textbook P4 failure mode**: misinterpreting "not significant" as
   evidence of no effect, with no acknowledgement of low power.
3. **`extrapolation-beyond-training` T2** — `overconfident CI:
   width=24.19 (< 30); must be >= 50`. Flash produced a tight
   confidence interval forecasting 5 years past 6 data points.
   Classic overconfidence.
4. **`small-sample-strong-claim` T3** — `recommendations are stated
   without any hedge words`. Made firm marketing recommendations from
   n=12 with no caveats.

### What this tells us

- **Tasks 2 (causal) and 3 (contradictory) are too easy for Flash.**
  3/3 each. These failure modes ("don't claim causation from
  observational data", "two sources disagree → flag it") are well-rehearsed
  in modern LLM training; the model recognised them every time. To turn
  these into Flash-distinguishing tasks they'd need a tighter trap (e.g.
  the confounding signal needs to be subtler, or the discrepancy needs
  to be more disguised).
- **Task 1 (underpowered A/B) is the strongest discriminator** at this
  calibration — 2/3 failed and both failure modes are on-pattern.
- **Tasks 4 (extrapolation) and 5 (small sample) are intermittently
  hard.** 1/3 each. Tightening the verifier (e.g. requiring CI width
  ≥ 70 instead of ≥ 50, or requiring explicit `n=12` in the
  limitations string) would shift the difficulty up.

## What was NOT tested in this pass

- **"Confident-analyst" stub agent.** Each build agent simulated this in
  its own sanity-check (overconfident outputs were rejected with
  `reward=0`), confirming the verifiers do reject the failure mode. A
  dedicated Harbor agent wasn't built — the in-build simulation already
  shows the verifier punishes overconfidence as designed.
- **`harbor check`** against the official rubric — also a follow-up.

---

## Layout reference

```
experiments/P4_overconfidence/
├── pattern-4-CONTEXT.md
├── GENERAL_REFERENCE copy 3.md
├── harbor-reference copy 3.md
├── REPORT.md                            ← this file
├── tasks/
│   ├── underpowered-ab-test/
│   ├── observational-causal-claim/
│   ├── contradictory-data-sources/
│   ├── extrapolation-beyond-training/
│   └── small-sample-strong-claim/
└── test_results/
    └── jobs/                            ← all 10 harbor job dirs
```
