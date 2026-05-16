# P3 Implicit-Constraints — Build & Test Report (v2)

**Pattern:** Tasks where the instructions describe a normal data-science deliverable, but the data itself signals a critical constraint a senior practitioner would notice. A literal instruction-follower misses the constraint and produces a wrong answer.

**Date:** 2026-05-16
**Harbor version:** 0.7.0
**Schema:** 1.2

---

## 1. v1 → v2 — why a redesign was needed

The v1 build (10/10 oracle/nop sanity cells passing) failed the actual goal of the eval: Gemini-3-flash-preview solved every task. Root cause: **the output-schema fields each verifier required were themselves hints**. Asking the agent to populate a `split_method` field telegraphs that the split method is a decision worth thinking about; asking for a `limitations` string prompts the agent to flag survivorship bias.

v2 makes each task **constraint-agnostic**: the schema looks identical whether the agent noticed the implicit constraint or not, and the verifier infers from the reported values (or from the agent's behavior on a hidden holdout) whether the constraint was handled.

### v1 → v2 diff summary

| # | v1 hint-y field | v2 verification approach |
|---|-----------------|--------------------------|
| T1 | `split_method` ("temporal"/"random") | Dropped. Verifier accepts `accuracy ∈ [0.65, 0.76]` only — random-split agents report ~0.80 → reject |
| T2 | `top_3_features` (importance ranking) | **Task replaced** (OLS is scale-invariant so units mismatch can't be verified through RMSE). New constraint: `zip_code` and `school_district` are integer-coded *categorical* columns with non-monotone price effects. Naive-numeric agents get RMSE ~3.7× the one-hot baseline → reject |
| T3 | `feature_list` (columns used) | Dropped. Verifier accepts `accuracy ∈ [0.73, 0.84]` only — leaky-cols agents report ~0.98 → reject |
| T4 | `limitations` field in JSON | Dropped. Output is free-form `/output/analysis.md`. Verifier requires BOTH (a) regression-analysis content AND (b) survivorship/selection-bias language. Caveat-dropping without analysis fails, and analysis-only without caveats fails |
| T5 | `performance_metric` (name + value separate fields) | Dropped entirely. Agent saves predictions on a held-out test set (`/workspace/holdout.csv`, no target visible). Verifier computes F1 itself from `/tests/holdout_labels.csv` (hidden) and requires F1 > 0.15 with non-constant predictions |

---

## 2. Tasks built (v2)

```
tasks/<name>/
├── instruction.md       # generic ML deliverable, no constraint hints
├── task.toml            # schema 1.2
├── generate_data.py     # seeded synthetic data generator
├── environment/         # Dockerfile + visible data files
├── tests/               # test.sh + verify.py (+ hidden ground truth for T5)
└── solution/            # solve.sh + solve.py (oracle reference)
```

| # | Task | Implicit constraint the model must infer |
|---|------|------------------------------------------|
| T1 | `p3-temporal-leakage-from-random-split` | `date` column + drifting target rate → random split leaks future info |
| T2 | `p3-units-mismatch-across-columns` * | `zip_code` and `school_district` are integer-coded categoricals, not numeric |
| T3 | `p3-target-leakage-from-column-name` | Two feature columns are post-outcome leaks of the target |
| T4 | `p3-survivorship-bias-in-dataset` | Dataset only contains companies that survived to 2024 (README discloses sampling source) |
| T5 | `p3-class-imbalance-not-mentioned` | Target is 97/3 — without class weighting, the model predicts all-zeros |

*T2 folder name kept for stability; content replaced (see §1).

---

## 3. Harbor test matrix (v2)

For each task we ran two Harbor jobs against the redesigned task:

- **`oracle`** — executes `solution/solve.sh` (the correct reference solution). Should reward 1.
- **`nop`** — does nothing. Should reward 0 (confirms the verifier doesn't pass on empty `/output`).

### Results — all 10 sanity cells pass

| # | Task | oracle | nop | Oracle verifier reason |
|---|------|:-----:|:---:|-------------------------|
| T1 | temporal-leakage-from-random-split | **1** ✓ | **0** ✓ | `accuracy=0.7165 (in [0.65,0.76]), model_type='LogisticRegression'` |
| T2 | units-mismatch-across-columns      | **1** ✓ | **0** ✓ | `RMSE=15,382 <= 1.3 × baseline 19,997` |
| T3 | target-leakage-from-column-name    | **1** ✓ | **0** ✓ | `accuracy=0.7930 within [0.73, 0.84]` |
| T4 | survivorship-bias-in-dataset       | **1** ✓ | **0** ✓ | `analysis keywords: 10/10, bias keywords: 5/13` |
| T5 | class-imbalance-not-mentioned      | **1** ✓ | **0** ✓ | `F1=0.1730 > 0.15` |

All five `nop` runs fail at "Missing /output/*" — agent produced no output to verify.

---

## 4. Constraint-agnostic instruction wording (v2)

Each instruction is written as a generic, realistic deliverable. The forbidden-word lists for each task were verified empty before dispatch.

**T1 — Customer Conversion Prediction.** "Train a model … When you report your test accuracy, it should reflect the accuracy you would expect this model to achieve when deployed on future customer data." No mention of "temporal", "chronological", "date", "split method", "drift".

**T2 — Property Price Prediction.** "Train the model … Hold out 20% as a test set and report the model's test RMSE." Schema: `{rmse, model_type}`. No mention of "categorical", "one-hot", "dummy", "zip code", "integer codes".

**T3 — Customer Churn Prediction.** "Hold out 20% of the data as a test set and report the model's test accuracy." Schema: `{accuracy, model_type}`. No mention of "leakage", "drop columns", "post-outcome", "examine columns".

**T4 — Predictors of Company Growth — Analysis Request.** "Read the data, perform an appropriate analysis (a regression is a reasonable starting point), and write a short analyst report — `/output/analysis.md` — describing your findings. Include the predictors and effect sizes you identified, **and anything else you think the partners should know before acting on this analysis.**" No mention of "limitations", "caveats", "bias", "survivorship", "selection", "failed companies".

**T5 — Classifier Training and Prediction.** "Train a binary classifier on `train.csv` … Then, for every row in `holdout.csv`, predict the target value (0 or 1) and write the predictions to `/output/predictions.csv` … Build the best classifier you can." No mention of "imbalance", "F1", "AUC", "metric", "class_weight", "threshold", "SMOTE".

---

## 5. Data-design numbers (v2)

| # | Rows | Generator validation diagnostics |
|---|-----:|----------------------------------|
| T1 | 10,000 | Random-split accuracy 0.8065 (inflated, fails verifier) vs. temporal-split 0.7165 (realistic, passes). Gap +0.090 |
| T2 | 5,000 | NAIVE (zip_code/school_district as numeric) RMSE = 56,584. AWARE (one-hot) RMSE = 15,382. **Ratio 3.68×**. Threshold = 1.3 × baseline = 19,997 |
| T3 | 5,000 | With leaky cols: accuracy 0.999 (fails high). Without: 0.793 (passes). Churn base rate 34.5% |
| T4 | 2,000 | Founding-year distribution: 2010→29, 2014→76, 2018→141, 2024→349 (sparse-old cohort). OLS R²=0.567 |
| T5 | 7,000 train / 1,000 holdout | Class balance 97/3 in both splits. Balanced LogReg F1 on holdout = 0.1730 (passes). Default LogReg → all-zeros predictions → constant-prediction check rejects |

---

## 6. Notable design notes (v2)

- **T1 row shuffling.** Generated rows are randomly shuffled before saving so an agent cannot trivially do "use first 80% as train". They must explicitly look at the `date` column to discover the drift.
- **T2 verifier baseline.** The verifier *computes* the AWARE baseline RMSE inside the container each run (no hard-coded number) — this makes the 1.3× threshold robust to any reasonable train/test split the agent chooses. Naive RMSE is 3.7× baseline so the discriminator is robust.
- **T4 dual-gate verifier.** Either failure mode is rejected: regression-only-no-caveats fails the bias-keyword check (proves the agent missed the survivorship signal), and caveats-only-no-analysis fails the analysis-content check (prevents the agent from gaming by dumping bias buzzwords without doing the work).
- **T5 hidden ground truth.** `/tests/holdout_labels.csv` exists only at verifier time. The agent's `/workspace/holdout.csv` has features + `id` but no target — they cannot peek. The verifier joins by `id`, requires no missing predictions, rejects constant predictions, and computes F1.
- **T5 model-choice sensitivity.** The minority-class signal is weak enough that *Random Forest* with `class_weight='balanced'` also predicts all-zeros. Only LogReg/linear models with balancing clear the F1 threshold. This is intentional — the test rewards both noticing the imbalance *and* choosing an appropriate handling approach, not just toggling a flag on any model.

---

## 7. Why this redesign is sound

The v1 verifiers were really testing "can the model fill in a structured form?", which any frontier model can do. The v2 verifiers test **behavior on the actual modeling task**:

- T1 catches inflated accuracy from leaked future data.
- T2 catches degraded RMSE from misencoded categoricals.
- T3 catches near-perfect accuracy from leaked target columns.
- T4 catches reports that omit the survivorship caveat *and* reports that are pure caveat without analysis.
- T5 catches predictions that collapse to all-majority (or get F1 < 0.15).

A model that doesn't notice the implicit constraint will fail on the metric, not on the schema. A model that flags the constraint but produces garbage will also fail (the verifiers all require evidence of actual work). This is what the take-home brief asked for: tasks Gemini-flash will fail at consistently.

---

## 8. Gemini-3-flash-preview live results (3 trials each)

Ran `harbor run -a gemini-cli -m google/gemini-3-flash-preview -k 3` against each task. Take-home target is pass@3 < 30% (i.e., model fails at least 3/3 trials).

| # | Task | Trial 1 | Trial 2 | Trial 3 | Pass rate | pass@3 | Take-home target met? |
|---|------|:-------:|:-------:|:-------:|:---------:|:------:|:--:|
| T1 | temporal-leakage-from-random-split | 1 | 1 | 1 | 3/3 | 1.0 | ✗ |
| T2 | units-mismatch-across-columns (categorical-as-numeric) | 1 | 1 | 0 | 2/3 | 1.0 | ✗ |
| T3 | target-leakage-from-column-name | 1 | 1 | 0 | 2/3 | 1.0 | ✗ |
| T4 | survivorship-bias-in-dataset | **0** | **0** | **0** | **0/3** | **0.0** | **✓** |
| T5 | class-imbalance-not-mentioned | 1 | 1 | 0 | 2/3 | 1.0 | ✗ |

**T4 is the only task that consistently catches Gemini-3-flash.** Failure modes across trials:
- Trial 1 (`__SRBnAbQ`): didn't produce `/output/analysis.md` at all
- Trial 2 (`__MBMkGxS`): produced an analysis.md, but fewer than 2 of the required analysis-keywords matched → "agent did not perform an analysis"
- Trial 3 (`__xkZV2df`): produced a proper analysis, but never flagged the survivorship bias — none of the 13 bias-awareness keywords matched

The other 4 tasks discriminate on individual trials but not consistently:
- **T1** — Gemini used `RandomForestClassifier` with identical accuracy=0.7325 across all 3 trials. The accuracy band [0.65, 0.76] was calibrated for LogReg-on-random-split (~0.81); RF appears more robust to drift, so even a random-split RF lands inside the band. **T1's verifier band fails to discriminate when the agent chooses RF.** This is a real flaw in the eval, not a real success for the model.
- **T2** — 1/3 trials Gemini treated zip_code/school_district as numeric (RMSE=56,898 vs. 1.3× baseline=19,997) → reject. The other 2 trials it one-hot encoded.
- **T3** — 1/3 trials Gemini kept the leaky columns → accuracy=1.0000 → reject.
- **T5** — 1/3 trials Gemini's model got F1=0.13 (below 0.15 threshold), the others got F1=0.20–0.21.

### Trial-level rewards

```
T1 temporal-leakage:        [1, 1, 1]  — all three Gemini runs used RandomForest, acc=0.7325
T2 categorical-as-numeric:  [1, 1, 0]  — fail at RMSE 56,898 (treated codes as numeric)
T3 target-leakage:          [1, 1, 0]  — fail at acc=1.0000 (kept leaky cols)
T4 survivorship-bias:       [0, 0, 0]  — missing analysis.md / no analysis / no bias acknowledgement
T5 class-imbalance:         [1, 1, 0]  — fail at F1=0.1304
```

---

## 9. Status & next steps

- ✅ All 5 v2 tasks built per the redesign plan
- ✅ All 5 oracle runs return reward 1 via Harbor (sanity)
- ✅ All 5 nop runs return reward 0 via Harbor (sanity)
- ✅ Gemini-3-flash-preview live battery complete (3 trials × 5 tasks = 15 trials)
- ✅ **T4 (survivorship-bias) meets the take-home target of pass@3 < 30%** (Gemini scored 0/3)
- ⚠️ T1's accuracy-band verifier doesn't bind when the agent picks RandomForest — needs a tighter discriminator (e.g., explicit model-type rejection, or move to a model-agnostic check via held-out chronological test data)
- ⚠️ T2/T3/T5 each caught Gemini once-in-three — they have *some* discriminating power but not enough for "consistent failure"
