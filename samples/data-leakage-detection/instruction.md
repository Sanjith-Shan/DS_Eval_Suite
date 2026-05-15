# Data Leakage Detection

The file `/workspace/pipeline.py` contains a training script for a customer-churn classifier. It currently reports a test accuracy around 97%, which the team believes is unrealistically high for this dataset (`/workspace/data.csv`).

Audit the pipeline. Identify every source of data leakage between train and test, then write a corrected version to `/output/pipeline_fixed.py`. The fixed script must:

- expose the same `train_and_evaluate(data_path: str) -> float` function returning the held-out test accuracy as a float;
- read the same `/workspace/data.csv` (do not modify the dataset);
- produce a realistic test accuracy that no longer relies on leaked information;
- run end-to-end when executed directly (`python /output/pipeline_fixed.py`), printing `TEST_ACCURACY=<value>`.

Do not change the train/test split parameters (`test_size=0.2`, `random_state=42`, `stratify=y`). You may keep, modify, or drop any feature in the dataset.
