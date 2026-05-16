# Pattern 1 — Live Gemini Evaluation Report

**Date:** 2026-05-16
**Model:** `google/gemini-3-flash-preview` (via Harbor's `gemini-cli` agent)
**Suite:** 6 Pattern-1 tasks × 3 trials each = 18 total runs
**Cost:** ≈ $1.44 total (6,798,950 input tokens, 141,999 output tokens)
**Wall time:** ≈ 67 minutes
**Runner:** `_build/run_gemini_p1.sh` → `jobs/<task>-gemini/result.json`

## Headline numbers

- **pass@1 = 10/18 = 55.6 %**
- **pass@3 = 4/6 = 66.7 %**  (a task counts as pass@3 if any trial succeeded)
- **Pattern 1 confirmed on 2 of 6 tasks** (T5 and T6 returned 0/3)
- **Pattern 1 partially confirmed on 1 task** (T3 returned 1/3)
- **Pattern 1 NOT confirmed on 3 of 6 tasks** (T1, T2, T4 each returned 3/3)

## Per-task results

| #  | Task | pass@1 | pass@3 | runtime | tokens (in / out) | cost | Verdict |
|----|------|--------|--------|---------|-------------------|------|---------|
| T1 | mnar-imputation-destroys-signal              | **3/3** | ✓ | 14m 10s | 721,438 / 24,610     | $0.20 | Pattern broken — Gemini did the MNAR audit |
| T2 | onehot-rare-categories-overfit               | **3/3** | ✓ | 9m 34s  | 1,335,689 / 41,302   | $0.34 | Pattern broken — Gemini handled rare-cats |
| T3 | outlier-removal-kills-minority-class         | **1/3** | ✓ | 20m 28s | 896,490 / 15,148     | $0.20 | Mixed — 2/3 trials failed protocol |
| T4 | normalization-destroys-temporal-feature      | **3/3** | ✓ | 10m 31s | 765,484 / 18,579     | $0.19 | Pattern broken — Gemini clipped/handled temporal |
| T5 | multicollinearity-after-log-transform        | **0/3** | ✗ | 5m 47s  | 585,566 / 19,668     | $0.17 | **Pattern PROVEN** — identical fail across trials |
| T6 | deduplication-loses-valid-longitudinal-data  | **0/3** | ✗ | 6m 17s  | 2,494,283 / 22,692   | $0.33 | **Pattern PROVEN** — identical fail across trials |

## Failure-mode detail

### T5 (multicollinearity) — pattern textbook-perfect

All three trials produced **identical** verifier output:

```
X1_pvalue=0.004512  X2_pvalue=0.122284  X3_pvalue=0.000000  bp_pvalue=0.275492
reward=0 reason=X2 not significant: p=0.122284 >= 0.05
```

Interpretation: Gemini applied the named fix (log-transform Y → BP test passes at p = 0.275), then ran plain OLS and reported the resulting p-values. It never checked VIF, never noticed the X1↔X2 collinearity that the log-transform amplified to ~0.99, and never tried ridge / orthogonalisation / PCA. This output is **bit-for-bit identical to our `named_fix_only_multicol.py` stub** — Gemini converged on the surface fix three times in a row, deterministically.

This is the cleanest demonstration of Pattern 1 in the suite.

### T6 (deduplication) — pattern textbook-perfect

All three trials produced AUC = 0.6733:

```
reward=0 reason=auc=0.6733 < 0.72 (threshold=0.72)
```

Interpretation: Gemini avoided the obvious "drop duplicates" mistake (which would have produced AUC ~0.57) but still failed to engineer delta features. The 0.6733 number matches the "all visits, no delta features" branch from the self-test bonus check — Gemini kept the longitudinal structure but treated each visit as an independent row with raw lab values. The audit step (computing `lab_value_1[v] - lab_value_1[v-1]` within each patient) was missed all three times. Identical AUC across trials suggests a deterministic convergence on the same partial solution.

### T3 (outlier-removal) — mixed: 2 protocol failures, 1 pass

```
trial 6aRLoVV: reward=0  reason=missing /output/classify.py
trial AF84P5P: reward=0  reason=missing /output/classify.py
trial yAithQj: reward=1  reason=class1_recall=0.8400 >= 0.55
```

Two of three trials never wrote `/output/classify.py` (likely wrote it to `/workspace/` or stdout instead). One trial wrote the file with class-1 recall of 0.84, well over the 0.55 threshold. T3 doesn't cleanly demonstrate the audit-step failure; rather, it surfaces a protocol-following inconsistency. If the agent reliably wrote to the right path, this task would likely pass at a higher rate.

### T1, T2, T4 — pattern broken (passed 3/3)

The current `gemini-3-flash-preview` (Nov 2025 release) appears strong enough to:

- **T1:** detect MNAR by examining the missingness pattern against other features, add a missingness indicator, and use a non-degenerate imputation. Three independent trials all landed X3_coef inside [0.30, 0.60] with p < 0.05.
- **T2:** fix the OHE/leakage bug AND filter rare categories (most likely via `OneHotEncoder(min_frequency=10)` or by switching to a different encoding). Three trials all in the [0.72, 0.84] band.
- **T4:** apply train-only scaling AND handle the out-of-distribution temporal feature (clip, quantile transform, or bin). Three trials all in [0.75, 0.86].

These three tasks were designed against the v1-era hypothesis that Gemini Flash misses second-order steps that aren't themselves named techniques. That hypothesis no longer holds for these specific audit steps with `gemini-3-flash-preview`. Either:

1. The model has improved on these particular cases since v1's data-leakage-detection result (where it was 0/3), OR
2. The audit steps in T1/T2/T4 are too "named" by 2026 standards (MNAR diagnostics, rare-category filtering, and temporal feature handling are now well-trodden), and the pattern only surfaces when the audit step is genuinely off the beaten path (T5's VIF check after log-transform; T6's delta-feature engineering).

## What this tells us about Pattern 1

Pattern 1 ("Surface Fix, Missed Consequence") is **real but specific**. It doesn't trigger every time you stack a named technique on top of a routine audit. It triggers reliably only when:

1. The named fix is well-known (heteroscedasticity → log, deduplication for repeated IDs).
2. The audit step is genuinely off the standard chain — not "always check residuals" or "always look at distributions," but a domain-specific second-order consequence (multicollinearity emerging *because* of the log-transform; delta features being the actual signal that visit_number happened to reveal).
3. The model can solve the named problem perfectly, so confidence is high, and it stops there.

T5 and T6 hit all three. T1, T2, T4 don't — modern Gemini sees those audit steps as part of the standard workflow now.

## Honest caveats

- **n=3 is noisy.** A true pass-rate measurement would want n=10+. The 3/3 and 0/3 splits we see are at the extremes of the binomial, but with 3 trials we can't distinguish a true 100 % pass rate from an 80 % pass rate, or a true 0 % from a 10 %.
- **Two task tomls changed mid-eval.** Originally `allow_internet = false` (matching the CONTEXT template); flipped to `true` so the `gemini-cli` agent could bootstrap (matching the v1 sample-task precedent). No data-generation or verifier code changed.
- **T2 deviates from the CONTEXT mechanism** (its "buggy pipeline" evaluates on the training set rather than doing OHE-before-split; OHE-before-split is kept as a red herring). This was documented in the verification report. T2 still tests the same pattern — surface fix vs. audit step — but the surface fix here is "fix the eval bug" not "encode after split."
- **T4's verifier band was tightened by 0.01** (from CONTEXT's [0.74, 0.86] to [0.75, 0.86]) because the named-fix-only stub landed at 0.746 on the chosen LR model. Documented in the verification report.
- **T3's two failed trials were protocol failures**, not numeric failures. The audit-step pattern wasn't really tested on those trials.

## Suggested next moves

1. **Replace T1, T2, T4 with harder audit steps** if the goal is "tasks where current Gemini Flash fails reliably." Candidates:
   - T1: switch the named fix from "handle missing values" to "fit a regression and report significance" — then make the missingness MNAR-and-correlated-with-target so that even multiple imputation needs a missingness indicator to recover the right coefficient.
   - T2: replace the rare-categories audit with something less well-trodden, e.g., post-encoding feature-importance leakage from auxiliary text fields.
   - T4: replace the temporal-clipping audit with a more subtle consequence, e.g., model calibration breakage from class-conditional drift.
2. **Run n=10 trials on T5 and T6** to confirm the 0/3 → 0/N stability of the failure.
3. **Keep T5 and T6 as canonical Pattern-1 examples.** They are clean, deterministic, and the failure mode is exactly what the pattern predicts.
4. **Add `allow_internet = true` to the CONTEXT.md template** so future Pattern folders don't trip the same agent-bootstrap failure.

## File pointers

- Raw harbor results: `jobs/{T1..T6 task name}-gemini/result.json`
- Per-trial agent trajectories: `jobs/<task>-gemini/<trial>/agent/`
- Per-trial verifier output: `jobs/<task>-gemini/<trial>/verifier/{reward.txt,test-stdout.txt,output.log}`
- Runner script: `_build/run_gemini_p1.sh`
- Raw battery log: `results/gemini_run.log`
- Sanity-check verification of the suite itself (oracle + nop + named-fix-only): `results/verification_report.md`
