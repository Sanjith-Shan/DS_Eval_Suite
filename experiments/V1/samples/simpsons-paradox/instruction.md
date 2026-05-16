# Hospital Treatment Comparison

`/workspace/outcomes.csv` records 700 patient outcomes from a hospital trial
comparing two treatments. Columns:

- `patient_id`
- `severity` — `"mild"` or `"severe"`
- `treatment` — `"A"` or `"B"`
- `outcome` — `"success"` or `"fail"`

Which treatment is more effective? Justify your conclusion and write your
analysis to `/output/analysis.json` with the schema:

```json
{
  "better_treatment": "A" | "B",
  "aggregate_rate_a": <float>,        // overall success rate for A
  "aggregate_rate_b": <float>,        // overall success rate for B
  "stratified_rates": {                // success rate per severity x treatment
    "mild":   {"A": <float>, "B": <float>},
    "severe": {"A": <float>, "B": <float>}
  },
  "stratified_analysis": <bool>,       // did your analysis include severity stratification?
  "paradox_identified": <bool>,        // did you encounter a paradox between aggregate and stratified results?
  "explanation": "<string>"            // 1-3 sentence explanation of what's going on
}
```

`/output` must exist before you write the file.
