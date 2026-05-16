# Classifier Pipeline — Fix the Leakage Bug

The file `/workspace/data.csv` contains 5000 rows of customer records with 10 numeric features (`X0`–`X9`), a high-cardinality categorical column `city_name` (120 unique values, Zipf-distributed so most cities appear rarely), and a binary `target` column.

The pipeline at `/workspace/pipeline.py` trains a `RandomForestClassifier` to predict `target`. The model team flagged the output as suspicious: the pipeline consistently reports over 90% test accuracy, which is implausibly high given the dataset difficulty. Reviewing the code, they believe there is a data-handling bug causing inflated accuracy. Your task is to find and fix the bug so that the pipeline reports a realistic held-out test accuracy.

Write your fixed pipeline to `/output/pipeline_fixed.py`. The file must expose a function with the following signature:

```python
def train_and_evaluate(data_path: str) -> float:
    ...
```

The function should read the CSV from `data_path`, train a `RandomForestClassifier(n_estimators=200, random_state=42)`, and return the **test-set accuracy** as a float. Use `train_test_split(test_size=0.2, random_state=42, stratify=y)` for the split. The `city_name` column carries genuine predictive signal and must be included in the model — do not drop it. The function will be imported and called directly by the evaluation harness.
