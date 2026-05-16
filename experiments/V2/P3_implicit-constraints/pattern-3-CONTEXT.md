# Pattern 3: Implicit Constraints

## What this pattern tests

Flash-tier models follow explicit instructions well but fail on rules requiring inference. DABstep's central finding (arXiv 2506.23719) was that agents are "considerably more prone to fail when they face a task in which they need to follow rules implicitly mentioned." BLADE showed agents achieve decent precision on basic decisions but poor coverage, because they only make the decisions that are explicitly prompted.

The key difference from other patterns: here the instructions are reasonable and complete for a human reader, but they omit a critical constraint that the DATA ITSELF signals. A senior data scientist would notice the constraint from the data and adjust. The model follows the instruction literally and gets a wrong answer.

## Design recipe for this pattern

1. Write clear, reasonable instructions that describe a legitimate analytical task
2. Omit one critical constraint that is VISIBLE IN THE DATA but not stated in the instructions
3. The correct answer requires the agent to notice the constraint from the data and adjust its approach
4. The verifier checks whether the final output reflects the constraint being handled

The omitted constraint must be:
- Discoverable from basic EDA (summary statistics, distributions, value ranges)
- Something a practitioner would catch in the first 5 minutes of looking at the data
- Important enough that ignoring it produces a materially wrong result
- NOT something the instructions hint at (no winking, no "be careful about...")

## Harbor task structure

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

## task.toml template

```toml
schema_version = "1.1"

[task]
name = "task-name"
description = "Short description"
authors = ["Sanjith"]
keywords = ["data-science", "implicit-constraints", "data-awareness"]

[agent]
timeout_sec = 600

[verifier]
timeout_sec = 120

[environment]
allow_internet = false
cpus = 2
memory_mb = 4096
storage_mb = 2048
```

## Build 5 tasks, described below

### Task 1: temporal-leakage-from-random-split

**Implicit constraint:** The data has a timestamp column and the target variable's distribution shifts over time (concept drift). A random train/test split leaks future information into training.

**Instructions say:** "Split the data into train and test sets (80/20), train a classifier, and report test accuracy to /output/results.json."

**What the data signals:** There's a clear date column. Sorting by date and plotting the target rate shows it increasing from 15% to 35% over the time period. A random split mixes future and past data. A temporal split (train on first 80% chronologically, test on last 20%) produces a much harder but realistic evaluation.

**Data generation:** 10000 rows spanning 2 years. Target rate drifts linearly from 15% to 35%. Features have stable relationships with target but the base rate shifts. Random split accuracy: ~0.82 (inflated by future leakage). Temporal split accuracy: ~0.71 (realistic).

**Verifier:** Check accuracy is in [0.65, 0.76]. Random split produces ~0.82 and fails (too high, outside band). Temporal split produces ~0.71 and passes. Also check that a "split_method" field in output is "temporal" or "chronological" (not "random").

**Solution:** Notice the date column. Plot target rate over time. See the drift. Split chronologically. Report realistic accuracy.

---

### Task 2: units-mismatch-across-columns

**Implicit constraint:** Two measurement columns are in different units (meters vs feet). The data doesn't label units. The feature values' ranges make it obvious to a human who looks at summary statistics.

**Instructions say:** "Build a regression model predicting property_price from the given features. Report RMSE and the top 3 most important features to /output/results.json."

**What the data signals:** lot_size ranges from 2000-15000 (square feet). building_area ranges from 50-300 (square meters). A human looking at summary stats would notice the scale discrepancy and either convert to common units or investigate. The model treats them as comparable, which distorts feature importance (lot_size dominates purely due to scale, not importance).

**Data generation:** 3000 properties. lot_size in sq ft (mean ~7000). building_area in sq meters (mean ~150). True relationship: price = 200*building_area_sqft + 15*lot_size + other_features + noise. In raw units, lot_size has 50x the numeric range of building_area, so an unscaled model gives lot_size much higher importance despite building_area being the stronger predictor per unit.

**Verifier:** Check that building_area (or equivalent) is in top_3_features. Check RMSE is below a threshold achievable only with correct unit handling. Without unit conversion, lot_size dominates and building_area is ranked 5th+, failing. With conversion or proper scaling, building_area ranks 1st.

**Solution:** Examine summary stats. Notice scale discrepancy. Research or infer that building_area is in sq meters. Convert to common units (or use feature scaling). Report correct feature importance.

---

### Task 3: target-leakage-from-column-name

**Implicit constraint:** One feature column would not be available at prediction time. It's derived from the target or from information that only exists after the outcome occurs.

**Instructions say:** "You are given customer data. Build a model to predict customer churn. Report accuracy, precision, recall, and the model to /output/results.json."

**What the data signals:** A column called "account_closure_reason" has values like "price", "competitor", "service", and NaN. The NaN values correspond perfectly to non-churned customers. This column is a direct leak of the target. Similarly, "last_contact_before_churn_days" is only populated for churned customers.

**Data generation:** 5000 customers. Columns include tenure, monthly_charges, contract_type, payment_method, account_closure_reason, last_contact_before_churn_days, num_support_tickets, churned (target). The two leaky columns have 100% correlation with the target by construction.

**Verifier:** Check that accuracy is in [0.73, 0.84]. A model including leaky columns gets ~0.98 accuracy (fails, too high). A model excluding them gets ~0.79 (passes). Also check that neither "account_closure_reason" nor "last_contact_before_churn_days" appears in a reported feature_list field.

**Solution:** Examine each column's relationship with the target. Notice that account_closure_reason and last_contact_before_churn_days are perfectly correlated with churn. Recognize these as target leakage (information that wouldn't exist at prediction time). Drop them. Train on remaining features.

---

### Task 4: survivorship-bias-in-dataset

**Implicit constraint:** The dataset only contains companies that survived to the present. Conclusions about "what predicts success" are biased because failed companies are missing.

**Instructions say:** "Analyze the provided company dataset to identify which factors predict company revenue growth. Run a regression and report significant predictors to /output/analysis.json."

**What the data signals:** A README.md in the environment says the data was "collected from active companies in the 2024 business registry." All companies in the dataset are currently operating. There's a founding_year column showing companies from 2010-2024 but the older cohorts are suspiciously sparse (survivorship). The dataset has no failed companies.

**Verifier:** Check that a "limitations" or "caveats" field in the output JSON exists and contains reference to survivorship bias, selection bias, or the fact that failed companies are excluded. Check that the reported coefficients include a caveat about generalizability. A straightforward regression with no caveats fails. An analysis that acknowledges the bias and adjusts (or at least flags it) passes.

**Solution:** Notice the data source description in README. Notice the sparse older cohorts. Flag survivorship bias in the analysis. Optionally adjust by noting that conclusions only apply to surviving companies and that risk factors for failure cannot be assessed.

---

### Task 5: class-imbalance-not-mentioned

**Implicit constraint:** The target variable has extreme class imbalance (97/3). Using accuracy as the evaluation metric is misleading. The instructions don't mention class distribution or evaluation metrics.

**Instructions say:** "Build the best classifier you can for the provided dataset. Report your model's performance to /output/results.json with fields model_type, performance_metric, and performance_value."

**What the data signals:** The target column has 97% zeros and 3% ones. A model that predicts all zeros achieves 97% accuracy. Basic EDA (value_counts on target) reveals the imbalance immediately.

**Data generation:** 8000 rows, 10 features, binary target at 97/3 ratio. Features have real but weak signal for the minority class. Baseline accuracy (all-zeros): 0.97. A good model with proper handling (SMOTE, class weights, threshold tuning) achieves F1 ~0.45, AUC ~0.82, accuracy ~0.90.

**Verifier:** Check that performance_metric is NOT "accuracy" (should be F1, AUC-ROC, precision-recall AUC, or similar). Check that performance_value is reasonable for the chosen metric (e.g., F1 between 0.30 and 0.60, AUC between 0.70 and 0.90). If the agent reports accuracy of 0.97+, it failed. If it reports accuracy at all (even 0.90), it failed to recognize that accuracy is inappropriate.

**Solution:** Check target distribution. Notice 97/3 imbalance. Choose appropriate metric (F1 or AUC). Apply class-weight balancing or SMOTE. Report F1 or AUC, not accuracy.

---

## After building all 5 tasks

For each task:
1. Write generate_data.py with fixed random seed
2. Copy generated data into environment/
3. `harbor run -a oracle` -> reward 1
4. `harbor run -a nop` -> reward 0
5. Also test a "literal instruction follower" stub that does exactly what the instructions say without noticing the implicit constraint. This stub should get reward 0.
