# Training Program Effectiveness

The company enrolled employees who scored below the 25th percentile on a skills
assessment into a training program. Performance was measured before and after
the program. The data is in `/workspace/employee_training.csv` with columns:

- `employee_id` -- unique identifier
- `group` -- employee group designation
- `pre_score` -- skills assessment score before the program
- `post_score` -- skills assessment score after the program

Analyze the data to determine whether the training program significantly
improved employee performance.

Save your results to `/output/analysis.json` with this schema (every value
shown is a placeholder; substitute your own computed values):

{
  "training_effective": "<bool>",
  "p_value": "<value>",
  "mean_improvement": "<value>"
}
