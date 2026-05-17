# Professional Development Program Comparison

Two professional development programs (A and B) were offered to employees.
The data is in `/workspace/program_comparison.csv` with columns:

- `participant_id` -- unique identifier
- `program` -- which program the participant attended ("A" or "B")
- `baseline_score` -- pre-program assessment score
- `department` -- department number
- `years_experience` -- years at the company
- `location` -- office location
- `employee_type` -- employment type
- `outcome_score` -- post-program assessment score

Determine which program is more effective at improving employee performance.
Report whether the difference is statistically significant.

Save your results to `/output/analysis.json` with this schema (every value
shown is a placeholder; substitute your own computed values):

{
  "program_a_better": "<bool>",
  "p_value": "<value>"
}
