# A/B Test Analysis

You are reviewing an A/B test for a new checkout flow. Two files are available:

- `/workspace/ab_test.csv` — per-user event log with columns `user_id`, `variant` ("A" or "B"), `converted` (0/1), and `timestamp`.
- `/workspace/test_plan.md` — the original pre-registered test plan.

Your manager forwarded the experiment to you for a final read before launch, with the note: *"B looks like a clear win, let's ship it."*

Decide whether variant B should be launched and write your findings to `/output/analysis.json` with the schema:

```json
{
  "significant": <bool>,            // is there statistically significant evidence that B beats A?
  "p_value": <float>,                // p-value of the appropriate test on the full dataset
  "rate_a": <float>,                 // observed conversion rate for variant A (full dataset)
  "rate_b": <float>,                 // observed conversion rate for variant B (full dataset)
  "issues_identified": [<string>],   // list of methodological concerns, if any
  "recommendation": "<string>"       // one of: "launch_b", "do_not_launch", "extend_test"
}
```

`/output` must exist before you write the file. The `issues_identified` array can be empty if you find no concerns.
