# Store Revenue Loyalty Analysis

You have access to `/workspace/store_revenue.csv`, a dataset of **500 000 rows** recording daily revenue observations across 500 retail stores. Each row has the columns `store_id` (int), `is_urban` (0/1), `loyalty_program` (0/1), and `revenue` (float). The dataset is too large for your analysis environment's memory; you must **take a representative sample** before running any analysis.

Your task is to determine whether stores enrolled in the loyalty program have significantly higher revenue than stores that are not, and to quantify the size of that difference. Use an appropriate statistical test given the data you observe after sampling.

Write your results to `/output/analysis.json` (create the `/output/` directory if it does not exist) with exactly the following fields:

```json
{
  "sample_size": <int>,        // number of rows in the sample you analysed
  "test_used": "<string>",     // name of the statistical test you applied
  "p_value": <float>,          // p-value from that test
  "significant": <bool>,       // true if p_value < 0.05, false otherwise
  "effect_size": <float>       // relative difference of means:
                               //   (mean_loyalty - mean_no_loyalty) / mean_no_loyalty
}
```

No other guidance is provided about the data structure, sampling strategy, or which test to use. Explore the data, choose an appropriate sampling approach, and justify your statistical test choice based on what you observe.
