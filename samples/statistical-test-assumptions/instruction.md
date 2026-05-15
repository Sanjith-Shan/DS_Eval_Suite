# Statistical Test with Assumption Checks

A retail chain has surveyed customer satisfaction at four store locations. The
data is in `/workspace/satisfaction.csv` with columns `store_id` (A/B/C/D) and
`satisfaction_score` (positive numeric). Sample sizes vary by store.

The marketing team wants to know: **do customer satisfaction levels differ
significantly across the four locations, and if so, which stores differ from
which?**

Write your analysis to `/output/analysis.json` with the schema:

```json
{
  "assumptions_checked": <bool>,           // did you check the assumptions of the test you chose?
  "normality_violated": <bool>,             // is normality violated in one or more groups?
  "equal_variance_violated": <bool>,        // is equal-variance violated?
  "test_used": "<string>",                  // name of the test used for the omnibus comparison
  "test_p_value": <float>,                  // p-value from that test
  "post_hoc_test": "<string>",              // name of the post-hoc test used (empty string if omnibus was non-significant)
  "group_medians": {"A": <float>, "B": <float>, "C": <float>, "D": <float>},
  "significant_pairs": [<string>]           // pairwise differences in the form "X>Y" (median of X is significantly greater than median of Y)
}
```

`/output` must exist before you write. Express the significant pairs as
inequalities between store ids (e.g. `"D>A"`).
