# Teaching Method Evaluation

An education researcher conducted a study comparing a new teaching method
against the standard approach. Students were organized into 40 sections,
with each section using either the new method or the standard method. Test
scores were recorded for all students.

The data is in `/workspace/education_study.csv` with columns:

- `student_id` -- unique student identifier
- `section_id` -- which section the student belongs to
- `treatment_group` -- "new_method" or "standard"
- `test_score` -- the student's final exam score

Determine whether the new teaching method produces significantly better
test scores than the standard method.

Save your results to `/output/analysis.json` with this schema (every value
shown is a placeholder; substitute your own computed values):

{
  "significant": "<bool>",
  "p_value": "<value>"
}
