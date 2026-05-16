# Pattern 2: Cascading Multi-Step Errors

## What this pattern tests

Flash-tier models self-condition on their own prior outputs. Each wrong step raises the probability of the next step being wrong by ~22.7 percentage points (Toolathlon, arXiv 2602.19008). Tasks with 4-6 dependent steps expose this because early mistakes silently propagate and the model doesn't backtrack to verify intermediate results.

The key difference from Pattern 1: Pattern 1 is about missing a follow-up audit after a correct fix. Pattern 2 is about making a wrong choice at step N and then building all subsequent steps on that wrong foundation, never going back to check.

## Design recipe for this pattern

1. Create a task with 4-6 clearly dependent steps where each step's output feeds into the next
2. Make step 1-2 require a non-obvious judgment call (not a named problem, just a choice between reasonable approaches)
3. Make the wrong-but-plausible choice at step 2 produce outputs that LOOK correct but are subtly wrong
4. Make steps 3-6 amplify the error so the final output is clearly wrong
5. The verifier checks only the final output, not intermediate steps

The critical property: the wrong intermediate result must LOOK plausible. If step 2 produces an obvious error (NaN, crash, empty dataframe), the model will notice and fix it. The cascade only works when intermediate outputs look reasonable but are silently wrong.

## Harbor task structure (same as other patterns)

```
task-name/
├── instruction.md
├── task.toml               # schema_version = "1.1"
├── environment/
│   ├── Dockerfile
│   └── (data files)
├── tests/
│   ├── test.sh
│   └── verify.py
└── solution/
    └── solve.sh
```

Same rules: test.sh writes 1/0 to /logs/verifier/reward.txt. /tests mounted only for verifier. No ground truth in Dockerfile. Verifier checks final output substance.

## task.toml template

```toml
schema_version = "1.1"

[task]
name = "task-name"
description = "Short description"
authors = ["Sanjith"]
keywords = ["data-science", "cascading-errors", "multi-step"]

[agent]
timeout_sec = 900

[verifier]
timeout_sec = 120

[environment]
allow_internet = false
cpus = 2
memory_mb = 4096
storage_mb = 2048
```

Note: timeout is 900s (15 min) because multi-step tasks take longer.

## Build 5 tasks, described below

### Task 1: wrong-join-cascades-to-wrong-report

**Step chain:** Load 3 tables -> Join them -> Aggregate -> Compare to threshold -> Write report

**The trap:** Table A (transactions) and Table B (customers) should be joined with a LEFT join (some transactions have customer_ids not in the customer table due to deleted accounts). Table C (regions) joins to B on region_id.

The plausible-but-wrong choice is INNER join on A-B, which silently drops ~18% of transactions (the deleted-account ones). These dropped transactions are disproportionately from Q4 (accounts deleted in a year-end cleanup). The aggregation then shows Q4 revenue 23% lower than reality. The final report incorrectly identifies Q4 as underperforming.

**Data generation:** Table A (transactions): 10000 rows with transaction_id, customer_id, amount, date, quarter. Table B (customers): 7500 rows (some customer_ids in A have no match). Table C (regions): 50 rows. The 2500 unmatched transactions are concentrated in Q4 (60% of them). Q4's true total is $2.1M but INNER join shows $1.6M.

**Instruction:** "You are given three CSV files. Join them into a single analytical dataset, compute quarterly revenue by region, and identify any quarter that underperformed by more than 10% vs the annual average. Write your findings to /output/report.json with fields quarterly_revenue (dict), underperforming_quarters (list), and total_annual_revenue (float)."

**Verifier:** Check total_annual_revenue is within 5% of the true value ($8.4M). Check that Q4 is NOT in underperforming_quarters (it isn't actually underperforming). INNER join produces total ~$7.1M and flags Q4, failing both checks.

**Solution:** Use LEFT join for A-B. Note unmatched customer_ids. Aggregate correctly. Q4 is fine.

---

### Task 2: wrong-encoding-cascades-to-wrong-features-to-wrong-model

**Step chain:** Encode categoricals -> Select features -> Train model -> Evaluate -> Report best features

**The trap:** Dataset has an ordinal categorical "education_level" with values ["high_school", "bachelors", "masters", "phd"]. The plausible-but-wrong choice is one-hot encoding (default for categoricals). The correct choice is ordinal encoding because there's a natural order.

One-hot encoding creates 4 binary columns. Feature selection (mutual information) then picks "phd" as important but misses "education_level" as a continuous predictor. The model trained on one-hot features gets decent accuracy (~0.74) but the reported "top features" are wrong, and a model using ordinal encoding gets ~0.82.

**Data generation:** 5000 rows, 12 features. education_level has a strong monotonic relationship with the target (each level adds ~0.15 to the probability). One-hot encoding fragments this into 4 weak signals instead of 1 strong one.

**Instruction:** "You are given a dataset with mixed numeric and categorical features. Prepare the data for modeling, select the top 5 most important features, train a classifier, and report the results to /output/results.json with fields accuracy (float), top_features (list of 5 strings), and model_type (string)."

**Verifier:** Check accuracy >= 0.79. Check that "education_level" (or equivalent ordinal name) appears in top_features. One-hot encoding produces accuracy ~0.74 and lists "phd" instead of "education_level", failing both.

**Solution:** Recognize the ordinal nature of education_level. Encode as integers (0,1,2,3). Feature selection correctly identifies it as the #2 predictor. Model accuracy improves.

---

### Task 3: wrong-aggregation-level-cascades-to-wrong-trend

**Step chain:** Clean data -> Aggregate to analysis grain -> Compute trend -> Identify anomalies -> Report

**The trap:** Raw data is at the hourly level. The instruction asks for "daily patterns." The plausible-but-wrong choice is to aggregate to daily totals (sum). The correct choice is daily averages, because some days have more operating hours than others (weekends have 8 hours of data, weekdays have 16). Daily sums make weekdays look 2x higher than weekends even when the hourly rate is identical.

The trend analysis then incorrectly identifies weekends as anomalously low. The anomaly report flags every weekend as underperforming.

**Data generation:** 90 days of hourly sensor readings. Weekdays have 16 hours of data (6am-10pm), weekends have 8 hours (10am-6pm). The hourly rate is CONSTANT across all days (~50 units/hour). Daily sums show weekdays at ~800, weekends at ~400. Daily averages show both at ~50.

**Instruction:** "You are given hourly operational data for 90 days. Analyze daily patterns, identify any days or day-types that show anomalous behavior (more than 1.5 standard deviations from the mean), and write findings to /output/analysis.json with fields daily_summary (dict of date to value), anomalous_days (list), and anomaly_explanation (string)."

**Verifier:** Check that anomalous_days list has fewer than 5 entries (there are genuinely 2-3 anomalous days planted with actual spikes). Sum-based aggregation flags all 26 weekend days as anomalous, producing a list of 28+ entries and failing. Average-based approach correctly identifies only the 2-3 planted anomalies.

**Solution:** Notice varying hours per day. Aggregate as mean per day, not sum. Identify only the genuinely anomalous days.

---

### Task 4: wrong-date-parsing-cascades-to-wrong-seasonality

**Step chain:** Parse dates -> Compute monthly aggregates -> Fit seasonal decomposition -> Forecast -> Report peak months

**The trap:** CSV contains dates in mixed formats. Most are MM/DD/YYYY but ~15% are DD/MM/YYYY (from a European data source merged in). Pandas will parse ambiguous dates (where day <= 12) incorrectly and silently succeed. Dates like "03/07/2024" get parsed as March 7 when they should be July 3.

This shifts ~15% of data points to wrong months. Monthly aggregates become noisy. Seasonal decomposition finds a weaker seasonal pattern than actually exists. The forecast underestimates peak months.

**Data generation:** 3 years of daily sales, strong December peak and June trough. 85% of rows in MM/DD/YYYY, 15% in DD/MM/YYYY (marked in a "source" column with values "US" and "EU", but the instruction doesn't mention this). The EU dates where day <= 12 get misparsed by default pandas read_csv.

**Instruction:** "You are given 3 years of daily sales data. Perform seasonal decomposition, identify the peak and trough months, and produce a 3-month forecast. Write results to /output/forecast.json with fields peak_month (int 1-12), trough_month (int 1-12), forecast_next_3_months (list of 3 floats), and seasonal_strength (float, ratio of seasonal component amplitude to trend)."

**Verifier:** Check peak_month == 12 and trough_month == 6. Check seasonal_strength > 0.3. Wrong date parsing produces peak_month as 11 or 1 (smeared), trough_month as 5 or 7, and seasonal_strength < 0.2. Correct parsing produces exact peak/trough and strength > 0.4.

**Solution:** Notice the "source" column. Parse US and EU dates separately with explicit format strings. Verify by checking that no impossible dates exist (month > 12 after parsing). Then proceed with decomposition.

---

### Task 5: wrong-sampling-cascades-to-wrong-distribution-to-wrong-test

**Step chain:** Sample from large dataset -> Compute statistics -> Choose statistical test -> Run test -> Report conclusion

**The trap:** The full dataset has 500K rows. The instruction says to "take a representative sample for analysis." The plausible approach is random sampling, which works fine. BUT the data has a clustered structure (500 stores, 1000 rows each). Simple random sampling produces a sample where some stores have 50+ rows and others have 0. The distribution of the sampled data appears normal (CLT effect), so the agent chooses a parametric test.

The correct approach is stratified sampling (sample N rows per store). With stratified sampling, the within-store variance is visible, the distribution is clearly non-normal (bimodal due to urban vs rural stores), and a non-parametric test is appropriate. The parametric test on random sample gives p = 0.08 (not significant). The non-parametric test on stratified sample gives p = 0.003 (significant).

**Data generation:** 500K rows, 500 stores. Urban stores (200) have mean revenue ~$5000, rural (300) have mean ~$3000. Within-store variance is moderate. The question is "do stores in the loyalty program have higher revenue?" Loyalty program stores genuinely do have ~8% higher revenue, but the effect is masked by the urban/rural confound in a random sample.

**Instruction:** "You have a 500K-row dataset of store-level daily revenue. The dataset is too large for your analysis environment (memory limit). Take a representative sample, determine whether stores in the loyalty program have significantly higher revenue than non-loyalty stores, and write results to /output/analysis.json with fields sample_size (int), test_used (string), p_value (float), significant (bool), and effect_size (float)."

**Verifier:** Check significant == true. Check p_value < 0.05. Check that effect_size is between 0.05 and 0.15. Random sampling + parametric test typically produces p > 0.05 and significant = false. Stratified sampling + appropriate test produces p < 0.01 and correct effect size.

**Solution:** Stratified sample by store_id (e.g., 20 rows per store = 10K total). Notice bimodal distribution. Use Mann-Whitney U or permutation test. Control for urban/rural. Report significant effect.

---

## After building all 5 tasks

For each task:
1. Write generate_data.py with fixed random seed
2. Copy generated data into environment/
3. `harbor run -a oracle` -> reward 1
4. `harbor run -a nop` -> reward 0
5. Also test a "plausible-but-wrong-path" stub that makes the wrong early choice and follows through logically. This stub should get reward 0. If it passes, the cascade isn't strong enough.
