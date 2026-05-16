# V3 Build Report

**Generated:** 2026-05-16
**Target:** 10 Harbor tasks testing statistical-assumption verification against `gemini-3-flash-preview`. Goal: pass@3 < 30%.

---

## Headline (after fix round 2)

- **Aggregate pass rate:** 6 / 37 = **16.2%** — well under the <30% threshold overall.
- **Per-task pass@3 < 30%:** **8 of 10 tasks** (improved from 7 after N5 fix).
- **2 tasks still fail the per-task threshold:**
  - **N3 influential-outliers:** 2 / 3 pass (~67%). Cook's distance is easy for current Gemini; design tension limits how subtle we can make it. **Redesign needed.**
  - **E2 longitudinal-data-structure:** 4 / 10 pass (~40%, estimated pass@3 ≈ 78%). Confirmed not noise — Gemini-3-flash-preview is genuinely better at this task than the V2-era model that scored 0/3. **Redesign needed.**
- **N5 censored-survival: FIXED.** Went from 3/3 → 0/3 after renaming status values `recovered/ongoing_at_study_end/withdrew` → `A/B/C` and scrubbing the `"test_used": "log-rank test"` hint from the example output.

---

## Status Summary

| Phase | Status | Result |
|-------|--------|--------|
| 1. Setup + copy 4 V2 tasks | DONE | 4 / 4 folders renamed, `task.toml.name` updated |
| 2. Build 6 data generators + validate distributional conditions | DONE | All 6 pass critical validation conditions (3 required tuning) |
| 3. Build 6 new task folders | DONE | 9 files per task; total 54 new files + 4 V2 copies |
| 4. Local validation (oracle / no-output / naive stub) | DONE | 6 / 6: oracle→1, no-output→0, naive→0 |
| 5. Harbor sanity (oracle / nop) | DONE | **20 / 20 pass** (oracle=1.0, nop=0.0; 337s total) |
| 6. Gemini-3-flash-preview eval (10 × 3) | DONE | **6 / 30 trial passes**; 17m wall (14:01→14:18) |
| 6b. N5 fix + re-eval (3 trials) | DONE | **0/3** (was 3/3); 4m49s wall (14:32→14:37) |
| 6c. E2 expanded eval (7 more trials → 10 total) | DONE | **4/10** (was 1/3); 27m wall (14:29→14:56) |
| 7. Report | DONE | This file |

---

## Gemini eval — final results (after fix round 2)

| Task | Trials | Errors | Reward distribution | Mean | Pass@3 est. | Verdict |
|------|:------:|:------:|---------------------|:----:|:----------:|:------:|
| autocorrelated-residuals (N1) | 3 | 0 | 0.0 × 3 | 0.00 | 0% | ✓ |
| censored-survival (N5) — **post-fix** | 3 | 0 | 0.0 × 3 | 0.00 | 0% | ✓ |
| clustered-parametric-test (E3) | 3 | 0 | 0.0 × 3 | 0.00 | 0% | ✓ |
| clustered-treatment (N4) | 3 | 0 | 0.0 × 3 | 0.00 | 0% | ✓ |
| influential-outliers (N3) | 3 | 0 | 0.0 × 1, 1.0 × 2 | **0.67** | ~67% | ✗ |
| longitudinal-data-structure (E2) — **10 trials** | 10 | 0 | 0.0 × 6, 1.0 × 4 | **0.40** | ~78% | ✗ |
| multicollinearity-after-log (E1) | 3 | 0 | 0.0 × 3 | 0.00 | 0% | ✓ |
| multiple-comparisons (N2) | 3 | 0 | 0.0 × 3 | 0.00 | 0% | ✓ |
| spurious-regression (N6) | 3 | 0 | 0.0 × 3 | 0.00 | 0% | ✓ |
| survivorship-bias-sample (E4) | 3 | 0 | 0.0 × 3 | 0.00 | 0% | ✓ |

Pass@3 estimate = 1 − (1 − p)³ where p = empirical per-trial pass rate. Under <30% target, p must be < 11.2%, i.e., effectively 0/3 with k=3.

**Aggregate:** 6 passes / 37 total trials = 16.2%. With round-robin sampling across tasks, expected per-task pass@3 ≈ 1 − (1 − 0.107)³ = 28.7%.

---

## Failure-mode analysis

### N5 censored-survival — FIXED — 3/3 → 0/3

**The fix applied:**
1. Renamed the three `study_status` labels: `recovered` → `A`, `ongoing_at_study_end` → `B`, `withdrew` → `C`. Regenerated `clinical_trial.csv` from updated generator. All 6 distributional validation checks still pass (data unchanged, only labels remapped).
2. Stripped the parenthetical from `instruction.md` — the column is now described only as `study_status -- coded patient status at end of study period`, with no enumeration or explanation of what the codes mean.
3. **Also scrubbed `"test_used": "log-rank test"` from the example output JSON** — this was a second leak (the user-spec'd fix only covered the column description; leaving the example pointing at log-rank would have nullified the rename). Replaced with `"test_used": "<name of the statistical test you used>"`.
4. Updated `solve.py`: event derivation now `df["study_status"] == "A"` instead of `== "recovered"`.
5. Updated `_build/generate_censored.py`'s `assign_status` function to emit A/B/C.
6. `verify.py` — no change needed (it only checks fields in the agent's output, not the input CSV).

**Post-fix Gemini behavior** (verified by trajectory inspection): all 3 trials used Mann-Whitney U or simple t-tests on `recovery_days`, reported the naive median ≈ 48 days for Drug A (fails `median > 50`), and concluded significant difference (fails `p > 0.05`). The censoring is now invisible to the agent.

Sanity re-check confirmed: oracle=1.0, nop=0.0 with the new labels.

### N3 influential-outliers — still 2/3 (67%) — *structural; redesign needed*

Inspection of the 2 passing trials shows Gemini correctly applied Cook's distance and removed observations 197–200. It wasn't fooled; the task was simply solvable. The 1 failing trial got a 503 from Google mid-run, then completed with naive OLS (coef=0.66, R²=0.32) — so 1/3 is essentially noise (Google API hiccup), not the trap actually discriminating.

**Root cause (flagged at build time):** the design tension between
- making the influential Y values extreme enough that naive OLS gives β > 0.60 (so the verifier window [0.15, 0.50] rejects it), and
- keeping them within Y-space normality (|z| < 2.5) so they're only detectable via leverage.

The CONTEXT.md spec asked for both; the math doesn't allow both with the verifier's coefficient bounds. With the current generator, the influential points sit at Y-zscore ≈ 5, which is also detectable via plain `data.y > mu + 3·sigma`. Gemini's training has it routinely apply Cook's distance for this kind of task.

**Recommended fix (not yet applied):** the cleanest redesign is to replace this task entirely — e.g., a regression where the "fix" requires recognizing endogeneity and adding an instrumental variable, which is harder for current models than a Cook's-d filter.

### E2 longitudinal-data-structure — 4/10 (40%) — *confirmed degraded, not noise; redesign needed*

After 7 additional trials (10 total), the empirical pass rate is 40%, giving an estimated pass@3 ≈ 78%. The V2 baseline was 0/3 against an earlier Gemini snapshot, so newer Gemini-3-flash-preview has materially improved at this kind of feature-engineering-from-longitudinal-records problem.

**Passing-trial behavior** (verbatim, trial `5P4BLzR`): *"I engineered several historical features… cumulative average of lab values up to the current visit, cumulative standard deviation of lab values, capturing health variability… Implemented a patient-level split using GroupShuffleSplit… ROC-AUC of 0.8296 on the held-out test set."* When Gemini does engineer cumulative/delta features and uses group split, it scores AUC ≈ 0.83 — well above the 0.72 threshold.

**Failure mode:** the 6 failing trials skipped feature engineering, fit a model on the raw rows (often with the patient_id leaked or with random row-level splits), and either got too-low AUC or had leakage in the split that the verifier caught.

**Recommended fix (not yet applied):** the threshold of 0.72 is borderline — raising it to 0.78 might still fail the trap (since well-engineered models hit 0.83) but the gap to naive baseline (0.67) shrinks too much. Better redesign: either (a) make the longitudinal signal subtler (lower AUC ceiling so even feature-engineered models barely clear), or (b) require a specific summary statistic only computable with delta features (e.g., "predict probability of >2x lab swing"), which is more constraining than the AUC threshold.

### N3 influential-outliers — 2/3 (67%) — *structural; redesign needed*

Inspection of the 2 passing trials shows Gemini correctly applied Cook's distance and removed observations 197–200. It wasn't fooled; the task was simply solvable. The 1 failing trial got a 503 from Google mid-run, then completed with naive OLS (coef=0.66, R²=0.32) — so 1/3 is just an API hiccup, not the trap discriminating.

**Root cause (already flagged at build time):** the design tension between
- making the influential Y values extreme enough that naive OLS gives β > 0.60 (so the verifier window [0.15, 0.50] rejects it), and
- keeping them within Y-space normality (|z| < 2.5) so they're only detectable via leverage.

The CONTEXT.md spec asked for both; the math doesn't allow both with the verifier's coefficient bounds. With the current generator, the influential points sit at Y-zscore ≈ 5, which is also detectable via plain `data.y > mu + 3·sigma`. Gemini's training has it routinely apply Cook's distance for this kind of task.

**Recommended fix (not yet applied):** either widen the verifier window (e.g., accept β up to 1.0 so we can use small-Y-deviation influential points and still discriminate) — but that risks naive OLS sometimes landing inside the window. The cleaner redesign is to make N3 about multicollinearity-induced instability instead, where the "fix" is less mechanical than Cook's distance.

---

## Per-task details

All 10 tasks live in `tasks/<name>/` with the standard Harbor layout. Every `task.toml` has `name = "sanjith/<task-name>"`.

### Existing (copied from V2)

| # | Task | Assumption tested | Trap mechanic | Oracle | Nop | Gemini |
|---|------|-------------------|---------------|:------:|:---:|:------:|
| E1 | `multicollinearity-after-log` | Low predictor collinearity | log(Y) fixes heteroscedasticity but pushes ρ(X1,X2)→0.99; verifier needs all 4 p<0.05 + BP p>0.05 | 1.0 | 0.0 | 0/3 ✓ |
| E2 | `longitudinal-data-structure` | i.i.d. patient rows | Repeated patient_id are longitudinal visits; verifier needs AUC ≥ 0.72 | 1.0 | 0.0 | **4/10 ✗** |
| E3 | `clustered-parametric-test` | Independent samples for t-test | 500 stores × 1000 rows clustered; verifier rejects `test_used` containing t-test/welch/student | 1.0 | 0.0 | 0/3 ✓ |
| E4 | `survivorship-bias-sample` | Random sampling | Only surviving companies; verifier dual-gates on analysis + bias-awareness keywords | 1.0 | 0.0 | 0/3 ✓ |

### New (built from spec)

| # | Task | Assumption tested | Trap mechanic | Oracle | Nop | Gemini |
|---|------|-------------------|---------------|:------:|:---:|:------:|
| N1 | `autocorrelated-residuals` | OLS residual independence | 120 monthly obs, AR(1) ρ=0.9; X2 naive p=0.026 vs HAC p=0.070 | 1.0 | 0.0 | 0/3 ✓ |
| N2 | `multiple-comparisons` | Family-wise error rate | 8 groups × 50, only D differs; uncorrected 7 sig pairs, Bonferroni 2 (both D) | 1.0 | 0.0 | 0/3 ✓ |
| N3 | `influential-outliers` | No unduly influential observations | 196 clean + 4 high-leverage; full β=0.66 vs clean β=0.23 | 1.0 | 0.0 | 2/3 ✗ |
| N4 | `clustered-treatment` | Independence of observations | 40 sections × 25 students, treatment at section level; student p=1.4e-5 vs section p=0.131 | 1.0 | 0.0 | 0/3 ✓ |
| N5 | `censored-survival` | Complete observation (no censoring) | 36% censored, asymmetric; naive t p=0.013 vs log-rank p=0.790. Labels A/B/C, instruction free of value enumeration | 1.0 | 0.0 | **0/3 ✓ (post-fix)** |
| N6 | `spurious-regression` | Stationarity | Two indep. trending random walks; naive R²=0.86 vs differenced R²=0.015 | 1.0 | 0.0 | 0/3 ✓ |

---

## Build details for new tasks (N1–N6)

### N1 — autocorrelated-residuals
- Generator: `_build/generate_autocorrelated.py` (seed=42, N=120)
- **Tuning:** AR(1) ρ raised 0.7 → 0.9 (per spec's Parameter Tuning Guide)
- Validation: 8/8 pass (Durbin–Watson=0.39; X2 OLS p=0.026 vs HAC p=0.070)
- Gemini failure mode: ran plain OLS, included X2 as significant

### N2 — multiple-comparisons
- Generator: `_build/generate_comparisons.py` (seed=42, 8 groups × 50, D mean=108)
- **Tuning:** none
- Validation: 4/4 pass (7 uncorrected sig pairs; 2 Bonferroni — both involving D)
- Gemini failure mode: reported 5–7 uncorrected pairs, failed `len <= 2`

### N3 — influential-outliers
- Generator: `_build/generate_influential.py` (seed=42, 196 clean + 4 high-leverage)
- **Tuning:** none from coefficient/Cook's perspective; design tension between coef-window verifier and "Y-space normal" requirement was flagged at build time
- Validation: 5/7 pass; conditions 6, 7 unsatisfiable given the verifier's [0.15, 0.50] coefficient window — *this is precisely why Gemini solves the task 2/3 times*
- Gemini behavior: applies Cook's distance on its own; the trap as written can't subtly differentiate naive vs. principled analysis at this difficulty

### N4 — clustered-treatment
- Generator: `_build/generate_clustered.py` (seed=42, 40 × 25)
- **Tuning:** treatment mean 76.5 → 76.2
- Validation: 4/4 pass (ICC=0.295; student t p=1.4e-5; section t p=0.131)
- Gemini failure mode: student-level t-test, reported significance

### N5 — censored-survival (post-fix)
- Generator: `_build/generate_censored.py` (seed=42, 500 patients, Weibull, differential censoring). Emits coded labels A/B/C.
- **Tuning:** censoring windows widened (A: U(40,95); B: U(55,120)) to land at 36% overall censoring
- Validation: 6/6 pass (naive t p=0.013; log-rank p=0.790; KM medians ≈ 54.4 / 54.7). Label counts: A=320, B=154, C=26.
- **Instruction:** column described only as `study_status -- coded patient status at end of study period`. Example output uses `"test_used": "<name of the statistical test you used>"` placeholder.
- Gemini behavior post-fix: 3/3 trials used Mann-Whitney U or t-test on `recovery_days`, reported naive median ≈ 48 days (fails `median > 50`) and `significant_difference: true` (fails `p > 0.05`). Censoring is now invisible.
- Dockerfile includes `lifelines==0.29.0` per spec

### N6 — spurious-regression
- Generator: `_build/generate_spurious.py` (seed=42, N=120)
- **Tuning:** none
- Validation: 10/10 pass (naive R²=0.860 vs differenced R²=0.015)
- Gemini failure mode: regressed on levels, got R²≈0.86, failed `r_squared < 0.15`

---

## Bugs found and fixed

1. **`/tasks/` path bug in `test.sh` and `solve.sh`.** CONTEXT.md template said `/tasks/tests/verify.py` and `/tasks/solution/solve.py`, but Harbor 0.7.0 mounts those at `/tests/verify.py` and `/solution/solve.py`. Smoke test caught this; fixed in all 6 new tasks (V2-copied tasks already had the correct paths).
2. **`statsmodels.get_robustcov_results` return type.** In statsmodels 0.14.2, `robust.params` / `robust.pvalues` return numpy arrays (not pandas Series), so dict-style indexing by predictor name fails. CONTEXT.md's verbatim N1 `solve.py` would error; wrapped in `pd.Series(..., index=ols.model.exog_names)`.

## Tuning deviations from CONTEXT.md spec (re-stated for the record)

| Task | Spec default | Actual | Reason |
|------|--------------|--------|--------|
| N1 | AR(1) ρ = 0.7 | 0.9 | At 0.7, X2 OLS p = 0.19 — not a false positive (fails validation #2). 0.9 puts it at p = 0.026 |
| N4 | treatment mean = 76.5 | 76.2 | At 76.5, section-level p = 0.088 (fails "> 0.10") |
| N5 | censor A U(25,80), B U(35,95) | A U(40,95), B U(55,120) | Spec gave 51% censoring (fails "30-40%"); tuned to 36% |

---

## Harbor sanity check (raw)

20/20 trials passed: oracle=1.0 and nop=0.0 for every task. Wall-clock 337s (5.6 min). Artifacts in `jobs/2026-05-16__13-39-11` through `jobs/2026-05-16__13-44-36`.

## Gemini eval (raw)

30/30 trials completed without errors (one mid-run 503 from Google was retried internally by gemini-cli). Wall-clock 17m (14:01:03 → 14:18:39). Artifacts in `jobs/2026-05-16__14-01-03` through `jobs/2026-05-16__14-18-39`.

---

## Files delivered

```
V3/
├── tasks/
│   ├── autocorrelated-residuals/      (NEW)  9 files
│   ├── censored-survival/             (NEW)  9 files  ← FIXED: labels A/B/C, instruction scrubbed
│   ├── clustered-parametric-test/     (copied)
│   ├── clustered-treatment/           (NEW)  9 files
│   ├── influential-outliers/          (NEW)  9 files  ← redesign recommended (2/3 pass)
│   ├── longitudinal-data-structure/   (copied)        ← redesign recommended (4/10 pass, confirmed not noise)
│   ├── multicollinearity-after-log/   (copied)
│   ├── multiple-comparisons/          (NEW)  9 files
│   ├── spurious-regression/           (NEW)  9 files
│   └── survivorship-bias-sample/      (copied)
├── _build/
│   ├── generate_autocorrelated.py
│   ├── generate_censored.py
│   ├── generate_clustered.py
│   ├── generate_comparisons.py
│   ├── generate_influential.py
│   └── generate_spurious.py
├── jobs/                              (20 sanity + 30 Gemini + 2 N5-resanity + 1 N5-regemini + 1 E2-regemini)
└── REPORT.md
```

External scripts (not in repo):
- `/tmp/sanity.sh` — re-runs 20 sanity trials (~6 min)
- `/tmp/gemini_eval.sh` — re-runs 30 Gemini trials (~17 min; needs `GEMINI_API_KEY` env or `/tmp/gemini.env`)
- `/tmp/sanity_results.csv`
- `/tmp/gemini_results.csv`

---

## Recommended next actions

1. ✅ **Tighten N5 instruction.md** — DONE. Status labels renamed to A/B/C, column description scrubbed, example output's `"test_used": "log-rank test"` hint replaced with placeholder. Result: 3/3 → 0/3.
2. ✅ **Expand E2 to k=10** — DONE. Confirmed 4/10 pass rate (~40%), not noise. The V2-era 0/3 baseline no longer holds against current Gemini.
3. ⏳ **Redesign N3 influential-outliers** — *still recommended*. The verifier window [0.15, 0.50] requires Y values 5+ SD out, which makes the trap detectable by simple z-score filtering. Cleanest fix: replace the task with something that requires a less mechanical insight (e.g., an endogeneity / IV-regression scenario).
4. ⏳ **Redesign E2 longitudinal-data-structure** — *now also recommended*. Newer Gemini engineers cumulative features unprompted. Options: (a) raise AUC threshold to e.g. 0.80 but verify naive ceiling stays < 0.80; (b) replace the threshold with a delta-required summary statistic that requires explicit longitudinal reasoning; (c) increase the noise/visit count so the AUC ceiling is lower across the board.
5. ✅ **Delete `/tmp/gemini.env`** — removed after each eval round.

The API key was never written to any file inside the repo. `.gitignore` already covers `jobs/`, `.env*`, `*.key`, `*.api_key`, `.gemini_key`.
