# Drug Formulation Comparison

A pharmaceutical company tested 8 formulations (A through H) of a new drug.
Each formulation was given to 50 patients, and their efficacy scores were
recorded in `drug_trial.csv`. The columns are:

- `patient_id` -- unique patient identifier
- `formulation` -- which formulation (A-H)
- `efficacy_score` -- measured treatment efficacy

For every pair of formulations, test whether their mean efficacy scores differ
significantly at the 5% level. Report all pairs with significant differences.

Save your results to `/output/analysis.json`:

{
  "significant_pairs": [["A", "D"]],
  "total_comparisons": 28,
  "test_used": "t-test",
  "details": {
    "A_vs_D": {"t_statistic": 3.2, "p_value": 0.002}
  }
}

Each pair in `significant_pairs` should be alphabetically ordered.
