# Clinical Trial Analysis

A clinical trial compared two drugs (Drug A and Drug B) for patient recovery.
The data is in `/workspace/clinical_trial.csv` with columns:

- `patient_id` -- unique patient identifier
- `drug` -- which drug the patient received ("drug_A" or "drug_B")
- `recovery_days` -- number of days until recovery
- `study_status` -- coded patient status at end of study period

Analyze the data and determine whether there is a significant difference
in recovery times between the two drugs. Report the median recovery time
for Drug A and whether the between-drug difference is statistically
significant.

Save your results to `/output/analysis.json` with this schema (every value
shown is a placeholder; substitute your own computed values):

{
  "median_recovery_drug_a": "<value>",
  "significant_difference": "<bool>",
  "p_value": "<value>"
}
