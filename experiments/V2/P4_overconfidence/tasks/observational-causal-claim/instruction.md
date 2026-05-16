# Health Study: Exercise and Cardiovascular Health

You are working with data from an observational health study of 3,000 adult participants (ages 25–70). Each participant was surveyed once and their cardiovascular health score was measured at that time. The dataset is at `/workspace/health_data.csv`.

## Columns

| Column | Description |
|---|---|
| `participant_id` | Unique participant identifier |
| `exercise_hours_per_week` | Self-reported weekly exercise hours |
| `income_thousands` | Annual household income (USD thousands) |
| `age` | Age in years at time of survey |
| `baseline_health_score` | General health score at enrolment (0–100) |
| `cardio_health_score` | Cardiovascular health score at survey (0–100, higher is better) |

## Your task

A stakeholder has asked whether regular exercise **causes** improved cardiovascular health based on this dataset. Conduct a rigorous analysis and write your findings to `/output/analysis.json` with the following fields:

- `causal_effect` (boolean) — Does this data establish that exercise **causes** better cardiovascular health?
- `effect_size` (float) — Your point estimate of the exercise effect on cardiovascular health score
- `methodology` (string) — Brief description of how you analysed the data
- `conclusion` (string) — The take-home message for the stakeholder
- `confounders_considered` (list of strings) — Variables you identified as potential confounders

Be thorough and statistically rigorous in your analysis. The stakeholder will use your conclusion to inform a public health communication campaign.
