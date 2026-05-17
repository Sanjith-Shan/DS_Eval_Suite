# Drug Formulation Comparison

A pharmaceutical company tested 8 formulations (A through H) of a new drug.
Each formulation was given to 50 patients, and their efficacy scores were
recorded in `/workspace/drug_trial.csv`. The columns are:

- `patient_id` -- unique patient identifier
- `formulation` -- which formulation (A-H)
- `efficacy_score` -- measured treatment efficacy

For every pair of formulations, test whether their mean efficacy scores differ
significantly at the 5% level. Report all pairs with significant differences.

Save your results to `/output/analysis.json` with this schema (every value
shown is a placeholder; substitute your own computed values):

{
  "significant_pairs": [["<letter>", "<letter>"], "..."],
  "total_comparisons": "<int>"
}

Each pair in `significant_pairs` should be alphabetically ordered.
