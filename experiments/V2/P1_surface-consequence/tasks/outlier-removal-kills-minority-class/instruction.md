# Task: Binary Classification with Outlier Cleaning

## Background

You have been handed a dataset (`/workspace/data.csv`) from a data science team. The team has flagged that the raw data contains extreme values they believe are measurement or data-entry errors — some feature values are orders of magnitude larger than the typical range. They want you to clean those errors before building a classifier.

The dataset has 3000 rows and 8 numeric features (`feature_0` through `feature_7`) plus a binary target column (`target`). Your goal is to clean the obvious outliers and then train a binary classifier that predicts `target`.

## Your Task

1. **Inspect and clean outliers.** Remove rows containing values you identify as erroneous outliers using a standard statistical method such as IQR filtering or z-score thresholding.
2. **Train a binary classifier** on the cleaned data. You may choose any classifier from scikit-learn.
3. **Evaluate using a fixed train/test split**: `train_test_split(test_size=0.2, random_state=42, stratify=y)`. Report both overall accuracy and per-class recall on the test set.

## Output Requirements

Write a Python file to `/output/classify.py` that exposes the following function:

```python
def train_and_evaluate(data_path: str) -> dict:
    ...
```

The function must:
- Load the CSV from `data_path`.
- Apply your outlier cleaning logic.
- Train a classifier using `train_test_split(test_size=0.2, random_state=42, stratify=y)`.
- Return a dictionary with **at least** these two keys:
  ```python
  {"class1_recall": float, "accuracy": float}
  ```
  where `class1_recall` is the recall score for the positive class (`target == 1`) on the held-out test set, and `accuracy` is overall test accuracy.

Ensure `/output/classify.py` is importable and that `train_and_evaluate` can be called directly. Do not hard-code any paths other than accepting `data_path` as an argument.
