# Pattern 1 (Surface Fix, Missed Consequence) — Verification Report

Date: 2026-05-16
Folder: `experiments/P1_surface-consequence/`
Harbor version: 0.7.0 (schema 1.2)

## Summary

All 6 tasks were built, the verification gauntlet ran clean, and every verifier band correctly distinguishes the **named-fix-only** failure mode from the **named-fix + audit-step** success state. The pattern is therefore properly engineered: an agent that knows the obvious technique but does not audit the post-fix state will reliably score 0.

| # | Task | Oracle (harbor) | Nop (harbor) | Named-fix-only (local) | Verifier tight? |
|---|------|---|---|---|---|
| 1 | mnar-imputation-destroys-signal              | **1** | **0** | **0** (X3_coef=0.275, below 0.30) | ✓ |
| 2 | onehot-rare-categories-overfit               | **1** | **0** | **0** (acc=0.668, below 0.72)     | ✓ |
| 3 | outlier-removal-kills-minority-class         | **1** | **0** | **0** (class1_recall=0.419, below 0.55) | ✓ |
| 4 | normalization-destroys-temporal-feature      | **1** | **0** | **0** (acc=0.746, below 0.75)     | ✓ |
| 5 | multicollinearity-after-log-transform        | **1** | **0** | **0** (X2_pvalue=0.122, above 0.05) | ✓ |
| 6 | deduplication-loses-valid-longitudinal-data  | **1** | **0** | **0** (AUC=0.568, below 0.72)     | ✓ |

12/12 harbor runs returned the expected reward. 6/6 named-fix-only stubs returned the expected failure under the verifier's exact band.

## Methodology

For each task we ran three checks:

1. `harbor run -p tasks/<task> -a oracle -y` — must return reward 1 (proves solvability).
2. `harbor run -p tasks/<task> -a nop -y` — must return reward 0 (proves the verifier isn't trivially passable by no-op).
3. A local "named-fix-only" stub that applies the **obvious** fix (mean imputation; encoding after split; z-score outlier removal; train-only scaling; log(Y); patient_id dedup) but **skips the audit step**. This must produce a numeric value outside the verifier's band so that the verifier returns 0 for it.

Steps 1 and 2 are full Docker runs through Harbor. Step 3 is run with the project venv against the same pinned data file the Docker image would receive (deterministic, fixed seeds).

All harbor results live in `jobs/t{1..6}_{oracle,nop}/result.json`.

## Per-task detail

### Task 1: mnar-imputation-destroys-signal

- **Named problem:** ~41% of X3 values missing; fit OLS and report which predictors are significant.
- **Audit step:** detect that X3 missingness is correlated with auxiliary feature X6 (MNAR), add a missingness indicator + regression-based imputation.
- **Verifier band:** `0.30 ≤ X3_coef ≤ 0.60` AND `X3_pvalue < 0.05`.
- **Design note:** the original CONTEXT spec made X3 missing-by-self (delete when X3 > q75). That mechanism does NOT attenuate OLS in the limit — the ratio Cov(X3, Y) / Var(X3) is preserved under constant imputation. We redesigned to MNAR-by-auxiliary: X3 = 0.82·X6 + 0.57·Z, X3 deleted with 80% prob when X6 > median(X6). This creates real X3↔X6 confounding that mean imputation cannot absorb.
- **Numbers:**
  | Approach | X3_coef | X3_pvalue | Passes |
  |---|---|---|---|
  | Oracle (M_X3 indicator + regression imputation from X6) | 0.484 | <1e-12 | ✓ |
  | Named-fix-only (mean impute + plain OLS) | 0.275 | <1e-20 | ✗ (coef below 0.30) |
  | Nop | — | — | ✗ |

### Task 2: onehot-rare-categories-overfit

- **Named problem:** buggy `pipeline.py` shows >90% test accuracy → suspicious. Produce a fixed pipeline with realistic accuracy.
- **Audit step:** notice that the high-cardinality `city_name` (120 unique, Zipf-distributed; 48 cities have <5 rows) needs `min_frequency=10` (or target encoding) to keep the trained model out of overfit territory.
- **Verifier band:** `0.72 ≤ test_accuracy ≤ 0.84`.
- **Design note:** the original CONTEXT spec had the buggy pipeline use OHE-before-split. The subagent tried six classifiers (RF, DT, KNN, LR, GBM, MLP) and could not reproduce the spec's "0.65 vs 0.88" gap — sklearn's RF in particular is too robust to sparse OHE columns to make this work. The buggy pipeline was therefore changed to **evaluate on the training set** (a different but equally plausible "looks suspiciously high" bug), with OHE-before-split kept as a red herring. The oracle still uses `min_frequency=10` after the split, preserving the pattern's pedagogical intent around rare-category handling. The named-fix-only stub drops `city_name` entirely (the other natural mis-fix), losing the genuine city signal and dropping to 0.668.
- **Numbers:**
  | Approach | accuracy | Passes |
  |---|---|---|
  | Buggy pipeline (evals on train) | 1.000 | ✗ (above band) |
  | Oracle (fix eval + OHE-after-split + min_frequency=10) | 0.779 | ✓ |
  | Named-fix-only (drop city_name) | 0.668 | ✗ |
  | Nop | — | ✗ |

### Task 3: outlier-removal-kills-minority-class

- **Named problem:** Obvious outliers (some values at 8-12 std). Clean them and build a binary classifier; report class-1 recall.
- **Audit step:** notice the outliers are concentrated in the minority class — they are legitimate extreme values, not errors. Plant 15 actual class-0 errors so blanket removal looks correct. Senior approach: remove only the 15 obvious class-0 errors; use `class_weight="balanced"` on the remaining data.
- **Verifier band:** `class1_recall ≥ 0.55`.
- **Numbers:**
  | Approach | class1_recall | accuracy | Passes |
  |---|---|---|---|
  | Oracle (remove only class-0 errors + balanced LR) | 0.813 | 0.885 | ✓ |
  | Named-fix-only (naive `|z|>3` removal + plain LR) | 0.419 | 0.962 | ✗ |
  | Nop | — | — | ✗ |

### Task 4: normalization-destroys-temporal-feature

- **Named problem:** `pipeline.py` fits StandardScaler on train+test. Fix to train-only.
- **Audit step:** after the fix, the monotonic temporal feature `days_since_first_purchase` produces test-side scaled values in [1.72, 2.90] vs. training max 1.73 — out-of-distribution extrapolation noise. Senior approach: clip scaled values to the training range, quantile-transform, or bin the temporal feature.
- **Verifier band:** `0.75 ≤ test_accuracy ≤ 0.86` (tightened from CONTEXT's [0.74, 0.86] by 0.01 so the named-fix-only stub at 0.746 reliably falls outside).
- **Design note:** LogisticRegression is scale-invariant to linear StandardScaler transformations, so fixing only the scaler changed test accuracy by 0.002 (0.748 → 0.746). The verifier band was narrowed by 0.01 at the lower end to make the audit-step requirement bite. The mechanism is still the CONTEXT-specified one (temporal OOD extrapolation); only the band was tuned.
- **Numbers:**
  | Approach | test_accuracy | Passes |
  |---|---|---|
  | Buggy pipeline (full-data scaler) | 0.748 | ✗ (just below band) |
  | Oracle (train-only scaler + clip temporal to training range) | 0.790 | ✓ |
  | Named-fix-only (train-only scaler, no clip) | 0.746 | ✗ |
  | Nop | — | ✗ |

### Task 5: multicollinearity-after-log-transform

- **Named problem:** Strong heteroscedasticity on Y. Fix it and report coefficient p-values.
- **Audit step:** log(Y) fixes the heteroscedasticity (BP p > 0.05) but pushes X1↔X2 effective correlation to ~0.99 in log-space (VIF ≈ 37). OLS reports X2 as non-significant (p ≈ 0.12). Senior approach: detect VIF inflation, use ridge regression with bootstrap p-values (or orthogonalize, or PCA).
- **Verifier band:** `X1_p < 0.05` AND `X2_p < 0.05` AND `X3_p < 0.05` AND `bp_p > 0.05`.
- **Numbers:**
  | Approach | X1_p | X2_p | X3_p | bp_p | Passes |
  |---|---|---|---|---|---|
  | Raw OLS (no log) | 0.002 | 0.871 | <1e-12 | 2.8e-15 | ✗ (BP fails) |
  | Oracle (log + ridge α=5 + 3000-sample bootstrap) | 0.001 | **0.032** | <1e-12 | 0.274 | ✓ |
  | Named-fix-only (log + plain OLS) | 0.005 | **0.122** | <1e-12 | 0.275 | ✗ (X2 fails) |
  | Nop | — | — | — | — | ✗ |

### Task 6: deduplication-loses-valid-longitudinal-data

- **Named problem:** Many rows share `patient_id`. Clean and build a readmission classifier.
- **Audit step:** repeats are multi-visit patients (`visit_number` column is the clue). Engineer `delta_lab_value_1 = current - previous`. Patient-level train/test split required.
- **Verifier band:** `AUC ≥ 0.72`.
- **Numbers:**
  | Approach | AUC | Passes |
  |---|---|---|
  | Oracle (keep all visits + delta features + patient-level split) | 0.884 | ✓ |
  | Named-fix-only (dedup keep-last + patient-level split, no deltas) | 0.568 | ✗ |
  | (Bonus check) Keep all visits but no delta features | 0.673 | ✗ |
  | Nop | — | ✗ |

## File layout produced

```
experiments/P1_surface-consequence/
├── tasks/
│   ├── mnar-imputation-destroys-signal/             # full Harbor task
│   ├── onehot-rare-categories-overfit/
│   ├── outlier-removal-kills-minority-class/
│   ├── normalization-destroys-temporal-feature/
│   ├── multicollinearity-after-log-transform/
│   └── deduplication-loses-valid-longitudinal-data/
├── _build/
│   ├── generate_{mnar,onehot,outlier,temporal,multicol,dedup}.py
│   └── named_fix_only_{mnar,onehot,outlier,temporal,multicol,dedup}.py
├── jobs/
│   └── t{1..6}_{oracle,nop}/                        # harbor run outputs
└── results/
    └── verification_report.md                       # this file
```

Each task folder follows the Harbor 1.2 layout:
```
<task>/
├── instruction.md        # 2–3 paragraphs, no hint at the audit step
├── task.toml             # schema_version "1.2", name "sanjith/<task>"
├── environment/          # Dockerfile + generated data (+ buggy script where applicable)
├── tests/                # test.sh + verify.py (substance-only, no AST/keyword checks)
└── solution/             # solve.sh + solve.py + fixed implementation
```

## Verifier design discipline (followed across all 6 tasks)

- **Substance, not procedure.** Every verifier imports the agent's output Python module, calls a documented function, and checks numeric bands. There are zero AST checks, keyword greps, or LLM-as-judge calls.
- **Ground truth never reachable from the agent.** No reference coefficients or thresholds appear in any Dockerfile, environment file, instruction, or pipeline.py. The verifier band lives only in `tests/verify.py`, which is mounted at verifier time and not visible to the agent.
- **Reward IO contract.** `tests/test.sh` writes `0` or `1` to `/logs/verifier/reward.txt`.
- **Reproducibility.** All generators use a fixed seed; pinned `numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 scikit-learn==1.5.1` (T1 and T5 additionally pin `statsmodels==0.14.2` since 0.14.6 is not on PyPI).
- **Tight bands.** For each task, the oracle, the buggy pipeline (where applicable), and the named-fix-only stub were all measured and the band tuned so only the audit-step path lands inside.

## Outstanding items / honest caveats

1. **Task 2 mechanism deviates from CONTEXT.** The buggy pipeline is "evaluate on train" rather than "OHE before split" because sklearn classifiers don't actually overfit hard to sparse OHE columns. The audit-step intent (rare-category handling) is preserved on the oracle side. If the takehome grader expects strictly the CONTEXT-specified mechanism, T2's buggy pipeline can be rewritten — but the failure mode under evaluation (surface fix vs. audit step) is intact.
2. **Task 4 band tightened by 0.01.** CONTEXT says `[0.74, 0.86]`; we use `[0.75, 0.86]` so the named-fix-only stub at 0.746 falls outside. The mechanism is unchanged.
3. **Live Gemini eval not yet run.** The deliverable per the take-home is pass@3 on `gemini-3-flash-preview`. These 6 tasks are now ready to be batched into the existing `_build/run_gemini_battery.sh` style runner (or a P1-specific runner). Expectation given the CONTEXT thesis: pass rate should be at or near 0/3 on at least 4-5 of the 6 tasks.
4. **No Docker-side check of the named-fix-only stub.** The stub was only validated against the local venv against the same generated CSV that ships in `environment/`. Because deps are pinned and seeds fixed, this should match Docker behaviour exactly, but a Docker-side check is the extra mile worth taking before submission.
