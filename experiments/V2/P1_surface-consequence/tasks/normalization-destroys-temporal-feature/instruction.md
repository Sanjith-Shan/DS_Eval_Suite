# Feature Scaling and Classification

The file `/workspace/data.csv` contains 4000 rows sorted chronologically. Each row has eight numeric features (`f0` through `f7`), one additional feature (`days_since_first_purchase`), and a binary target column (`target`). The dataset represents a classification problem where the first 3000 rows are the training set and the last 1000 rows are the test set (split by row index, not by shuffling).

The file `/workspace/pipeline.py` contains a working but flawed implementation. It trains a logistic regression classifier and evaluates test accuracy. The pipeline contains a feature-scaling bug: the `StandardScaler` is fit using the statistics of the **entire dataset** (all 4000 rows) rather than the training rows only. This is data leakage — information about the test distribution is incorporated into the scaler during training, which violates the train/test boundary.

Your task is to produce a corrected pipeline that achieves a test accuracy in the range **[0.75, 0.86]**. Write your solution to `/output/pipeline_fixed.py`. That file must expose a function with the following exact signature:

```python
def train_and_evaluate(data_path: str) -> float:
    ...
```

The function should read the CSV from `data_path`, apply the train/test split (first 3000 rows for training, last 1000 rows for testing), train a classifier, and return the test accuracy as a float. The function will be imported and called directly by the evaluation harness, so it must run end-to-end without side effects and without printing. The script should also be executable directly (`python /output/pipeline_fixed.py`), in which case it may print the accuracy.
