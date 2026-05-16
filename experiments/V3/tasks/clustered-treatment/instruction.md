# Teaching Method Evaluation

An education researcher conducted a study comparing a new teaching method
against the standard approach. Students were organized into 40 sections,
with each section using either the new method or the standard method. Test
scores were recorded for all students.

The data is in `education_study.csv` with columns:

- `student_id` -- unique student identifier
- `section_id` -- which section the student belongs to
- `treatment_group` -- "new_method" or "standard"
- `test_score` -- the student's final exam score

Determine whether the new teaching method produces significantly better
test scores than the standard method.

Save your results to `/output/analysis.json`:

{
  "significant": true,
  "p_value": 0.03,
  "effect_size": 2.5,
  "test_used": "independent two-sample t-test",
  "mean_new_method": 78.2,
  "mean_standard": 75.7
}
