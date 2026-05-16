# Clinical Trial Analysis

A clinical trial compared two drugs (Drug A and Drug B) for patient recovery.
The data is in `clinical_trial.csv` with columns:

- `patient_id` -- unique patient identifier
- `drug` -- which drug the patient received ("drug_A" or "drug_B")
- `recovery_days` -- number of days until recovery
- `study_status` -- coded patient status at end of study period

Analyze the data and determine whether there is a significant difference
in recovery times between the two drugs. Report the median recovery time
for each drug and whether the difference is statistically significant.

Save your results to `/output/analysis.json`:

{
  "median_recovery_drug_a": 58.0,
  "median_recovery_drug_b": 62.0,
  "significant_difference": false,
  "p_value": 0.35,
  "test_used": "<name of the statistical test you used>"
}
