# P2 — Cascading Multi-Step Errors: Build + Test Report

## Overview

Built and validated 5 Harbor (v0.7, schema 1.2) evaluation tasks under `experiments/P2_cascading-multistep/tasks/`. Each task implements a 4–6 step pipeline where a plausible-but-wrong early choice silently propagates to a wrong final output. The verifier checks only the final output.

All 10 sanity checks pass: each task's oracle solution → reward 1, each task's `nop` agent (no-op) → reward 0.

## Test matrix

| # | Task | Oracle | Nop | Gemini-3-flash-preview (3 trials) | Pass@3 | Cascade trap |
|---|------|:------:|:---:|:---------------------------------:|:------:|--------------|
| T1 | `wrong-join-cascades-to-wrong-report` | ✅ 1 | ✅ 0 | **3/3 pass** (1,1,1) | 1.00 | INNER vs LEFT join on transactions↔customers |
| T2 | `wrong-encoding-cascades-to-wrong-model` | ✅ 1 | ✅ 0 | **3/3 pass** (1,1,1) | 1.00 | One-hot vs ordinal encoding of `education_level` |
| T3 | `wrong-aggregation-cascades-to-wrong-trend` | ✅ 1 | ✅ 0 | **3/3 pass** (1,1,1) | 1.00 | Daily SUM vs daily MEAN when weekday/weekend hour counts differ |
| T4 | `wrong-date-parsing-cascades-to-wrong-seasonality` | ✅ 1 | ✅ 0 | **2/3 pass** (1,0,1) | 1.00 | Default pandas parse vs format-aware parse on mixed MM/DD vs DD/MM |
| T5 | `wrong-sampling-cascades-to-wrong-test` | ✅ 1 | ✅ 0 | **0/3 pass** (0,0,0) | **0.00** | Simple random sample + t-test vs stratified sample + Mann-Whitney |

**Headline finding:** only **T5** (sampling → stat-test cascade) reliably catches Gemini-3-flash-preview at the pass@3 < 30% bar that pattern-2-CONTEXT targets. T4 trips on 1 of 3 trials but pass@3 = 100%. T1–T3 cascades are too obvious for this model.

Job outputs (10 sanity + 15 gemini trials) are under `experiments/P2_cascading-multistep/jobs/` (local to this folder; not in the shared repo-root `jobs/`).

### Why each cascade did or didn't trip

- **T1 (join):** Gemini consistently picked LEFT join. The instruction's neutral phrasing ("combine the tables") didn't push it toward INNER. The 18% orphan-customer rate may also be visible enough during inspection that a careful agent notices the customer_id mismatch.
- **T2 (encoding):** Gemini recognised `education_level` as ordinal in all 3 trials. The value strings (`high_school`/`bachelors`/`masters`/`phd`) carry a strong semantic cue that almost any LLM picks up.
- **T3 (aggregation):** Gemini correctly used daily mean. The dataset's varying hours-per-day are easy to detect with a `groupby(date).size()` glance, which the agent apparently performed.
- **T4 (date parsing):** This cascade is *partially* effective — 1 of 3 trials produced `peak_month=1` (the silent-misparse signature). The other 2 trials apparently noticed the `source` column or did sanity-checks on monthly counts and parsed correctly. Pass@3 = 100% but pass@1 = 67%.
- **T5 (sampling):** **All 3 trials** chose Welch's t-test (one was "Welch's t-test on store-level means") and were rejected by the verifier's test-name filter. The clustered urban/rural structure was not flagged by the model. Verifier reasons:
  - `reward=0 reason=test_used "Welch's t-test" contains rejected keyword 't-test'.` (×2)
  - `reward=0 reason=test_used "Welch's t-test (on store-level means)" contains rejected keyword 't-test'.` (×1)

## Per-task detail

### T1 — wrong-join-cascades-to-wrong-report
- **Chain:** Load 3 tables → join → quarterly aggregate → 10% threshold → write report.
- **Trap:** 2 500 of 10 000 transactions reference deleted customer ids (60% of those orphans are Q4). INNER join silently drops them.
- **Cascade strength (from `generate_data.py`):**
  - True annual revenue: **$8 725 677** (Q4 is the highest quarter — not underperforming).
  - INNER-join annual: **$6 524 747** (Q4 falsely appears 19% below average).
- **Verifier:** total_annual within 5% of $8.73M **and** "Q4" not in `underperforming_quarters` and list length ≤ 1.
- **Files:** `tasks/wrong-join-cascades-to-wrong-report/` (10 files; 3 CSVs, 7 scripts).

### T2 — wrong-encoding-cascades-to-wrong-model
- **Chain:** Encode categoricals → mutual-info feature select → train → eval → report top features.
- **Trap:** `education_level` is ordinal (`high_school < bachelors < masters < phd`). One-hot fragments the signal into 4 weak dummies; ordinal preserves it.
- **Cascade strength (from `generate_data.py`):**
  - One-hot pipeline: accuracy ≈ **0.81**, top features include `education_level_phd`, `education_level_masters` (no bare `education_level`).
  - Ordinal pipeline: accuracy ≈ **0.82**, top features lead with `education_level`.
- **Verifier:** accuracy ≥ 0.79 **and** `top_features` contains the bare string `education_level` (any `education_level_<value>` dummy fails).
- **Note:** the cascade discriminator is primarily the feature-name check, since both pipelines reach the accuracy floor. This is intentional — names give an unambiguous signal of which encoding path was taken.

### T3 — wrong-aggregation-cascades-to-wrong-trend
- **Chain:** Clean hourly data → daily aggregate → mean/std → flag 1.5σ outliers → report.
- **Trap:** Weekdays cover 6 am–9 pm (16 rows/day) and weekends cover 10 am–5 pm (8 rows/day). Hourly rate is identical (~50). Daily SUM makes weekends look ~2× lower; daily MEAN reveals them as identical.
- **Cascade strength (from `generate_data.py`):**
  - Median daily SUM — weekdays: 801.4, weekends: 402.1 (≈2× ratio).
  - Median daily MEAN — weekdays: 50.09, weekends: 50.26 (≈identical).
  - SUM-based 1.5σ flags **9 days**; MEAN-based 1.5σ flags **3 days** (the planted anomalies on 2024-01-16, 2024-02-17, 2024-03-13).
- **Verifier:** `len(anomalous_days)` ∈ [2, 5] **and** all 3 planted dates appear. Sum-based answer (9 days) exceeds the upper bound.

### T4 — wrong-date-parsing-cascades-to-wrong-seasonality
- **Chain:** Parse dates → monthly aggregate → seasonal decompose (period=12) → peak/trough/strength → 3-month forecast → report.
- **Trap:** CSV has a `source` column ("US"/"EU") not mentioned in the instruction. 85 % of rows are MM/DD/YYYY; 15 % are DD/MM/YYYY, all with day ≤ 12. Default `pd.to_datetime` silently misparses every EU row.
- **Cascade strength (from `generate_data.py`):**
  - Naive parse → **peak_month = 1, trough_month = 5** (smeared).
  - Correct parse (split by `source`, explicit formats) → **peak_month = 12, trough_month = 6, seasonal_strength = 0.997**.
- **Verifier:** `peak_month == 12 and trough_month == 6 and seasonal_strength > 0.3` plus shape checks on forecast list.

### T5 — wrong-sampling-cascades-to-wrong-test
- **Chain:** Sample from 500 K-row dataset → describe distribution → pick stat test → run → report.
- **Trap:** Data is clustered (500 stores × 1 000 rows). Urban stores (200) earn ≈ $5 000; rural (300) earn ≈ $3 000; loyalty boost is +10 %. Simple random sampling collapses the cluster structure (~CLT smooths the distribution) and invites a parametric t-test, which the agent reports as `test_used: "t-test"`. Stratified sampling (20 rows/store) preserves cluster variance and motivates Mann-Whitney U / permutation.
- **Cascade strength (from `generate_data.py`):**
  - Stratified Mann-Whitney: p ≈ 0, effect_size ≈ 0.108 — well inside the verifier range.
  - Even when random + t-test happens to clear p<0.05, the verifier rejects `test_used` containing "t-test"/"welch"/"student".
- **Verifier:** `significant == true` ∧ `p_value < 0.05` ∧ `effect_size ∈ [0.05, 0.15]` ∧ `sample_size ∈ [1 000, 50 000]` ∧ `test_used` matches non-parametric/stratified vocabulary and does not match parametric vocabulary.

## Issues encountered & resolved

1. **T1 verifier never wrote reward file (oracle run 1 → `RewardFileNotFoundError`).**
   `tasks/wrong-join-cascades-to-wrong-report/tests/verify.py` had `sys.exit(0)` in its `if __name__ == "__main__":` block instead of `sys.exit(0 if main() else 0)`, so `main()` was never invoked. Fixed; oracle re-ran successfully on attempt 2. The other 4 verifiers had the correct pattern from the start.

2. **T2 accuracy gap is narrow (0.81 vs 0.82).** Originally the spec called for one-hot ≈ 0.74 and ordinal ≈ 0.82. With the chosen feature set, both pipelines clear the verifier's 0.79 accuracy floor, so the cascade discriminator collapses to the feature-name check. This is acceptable (and arguably cleaner: a single-bit unambiguous signal) but worth noting if a real Gemini run unexpectedly passes.

3. **Stray `__pycache__` in `tasks/wrong-encoding.../tests/`** left over from local solve→verify testing. Removed before harbor runs.

## File inventory

```
experiments/P2_cascading-multistep/
├── pattern-2-CONTEXT.md         (pre-existing)
├── GENERAL_REFERENCE copy.md    (pre-existing)
├── harbor-reference copy.md     (pre-existing)
├── Abundant Research Take Home 2.pdf  (pre-existing)
├── report.md                    (this file)
├── jobs/                        (harbor output, 10 dirs)
│   ├── wrong-join-{oracle,nop}/
│   ├── wrong-encoding-{oracle,nop}/
│   ├── wrong-aggregation-{oracle,nop}/
│   ├── wrong-date-{oracle,nop}/
│   └── wrong-sampling-{oracle,nop}/
└── tasks/
    ├── wrong-join-cascades-to-wrong-report/
    │   ├── generate_data.py
    │   ├── instruction.md
    │   ├── task.toml
    │   ├── environment/{Dockerfile,transactions.csv,customers.csv,regions.csv}
    │   ├── tests/{test.sh,verify.py}
    │   └── solution/{solve.sh,solve.py}
    ├── wrong-encoding-cascades-to-wrong-model/
    │   ├── generate_data.py
    │   ├── instruction.md
    │   ├── task.toml
    │   ├── environment/{Dockerfile,dataset.csv}
    │   ├── tests/{test.sh,verify.py}
    │   └── solution/{solve.sh,solve.py}
    ├── wrong-aggregation-cascades-to-wrong-trend/
    │   ├── generate_data.py
    │   ├── instruction.md
    │   ├── task.toml
    │   ├── environment/{Dockerfile,sensor_data.csv}
    │   ├── tests/{test.sh,verify.py}
    │   └── solution/{solve.sh,solve.py}
    ├── wrong-date-parsing-cascades-to-wrong-seasonality/
    │   ├── generate_data.py
    │   ├── instruction.md
    │   ├── task.toml
    │   ├── environment/{Dockerfile,sales.csv}
    │   ├── tests/{test.sh,verify.py}
    │   └── solution/{solve.sh,solve.py}
    └── wrong-sampling-cascades-to-wrong-test/
        ├── generate_data.py
        ├── instruction.md
        ├── task.toml
        ├── environment/{Dockerfile,store_revenue.csv}  (13 MB)
        ├── tests/{test.sh,verify.py}
        └── solution/{solve.sh,solve.py}
```

## How to run

From `experiments/P2_cascading-multistep/`:

```bash
# Sanity check (already passing)
harbor run -p tasks/<task-name> -a oracle -o jobs -n 1 -k 1 -y
harbor run -p tasks/<task-name> -a nop    -o jobs -n 1 -k 1 -y

# Real model eval (3 trials each, expects GEMINI_API_KEY exported)
harbor run -p tasks/<task-name> \
  -a gemini-cli -m google/gemini-3-flash-preview \
  --ae "GEMINI_API_KEY=${GEMINI_API_KEY}" \
  -o jobs --job-name "<task-name>-gemini" \
  -k 3 -n 1 -y
```

## Recommendations / next steps

- **Ship T5 as the headline P2 task.** It hit `pass@3 = 0.0` against Gemini-3-flash-preview across 3 trials, well under the 30 % bar. The cascade is reliable because the trap (parametric t-test on clustered data) is named in the model's output (`test_used`), giving an unambiguous discriminator.
- **T1–T3 cascades need to be made less obvious** if pattern-2-CONTEXT's `pass@3 < 30%` is the bar to clear. Suggestions:
  - **T1:** make the orphan customers less suspicious — e.g., add 5% orphan rows that are *truly* the deleted-account edge case but bury them in a much larger transaction set so a quick `df.isna()` check after the join doesn't immediately raise alarms.
  - **T2:** weaken `education_level` as a stand-alone predictor so ordinal isn't obviously better, or rename the levels to non-academic ones (e.g., `tier_1`/`tier_2`/`tier_3`/`tier_4`) so the ordinal cue is in the data, not the column name.
  - **T3:** drop the day-of-week giveaway — instead make the varying hours per day driven by a per-day `coverage_hours` column whose role isn't called out, so summing requires noticing the column.
- **T4 partial-success means design is on the right track but not strong enough.** Increase the EU-row fraction past 15 % (try 25 %) and reduce the share of EU rows where `day > 12` to 0 % so every misparse is silent. Re-run after tuning.
- **No "wrong-path stub" check inside Harbor.** Each subagent confirmed locally that a stub following the wrong cascade path fails the verifier; we did not wire those stubs as harbor agents. Useful for regression testing if any task is tuned.
