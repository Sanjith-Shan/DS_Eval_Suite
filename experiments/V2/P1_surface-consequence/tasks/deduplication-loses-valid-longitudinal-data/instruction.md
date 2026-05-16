# Readmission Risk Modeling

You have been given a patient dataset at `/workspace/data.csv`. The dataset contains records for 800 patients and includes the following columns:

- `patient_id` — unique patient identifier (string)
- `visit_number` — integer
- `days_between_visits` — integer (number of days since the previous record for this patient; 0 for the first record)
- `lab_value_1` — continuous float
- `lab_value_2` — continuous float
- `readmitted` — binary target (1 = readmitted, 0 = not)

You will notice that many rows share the same `patient_id`. Before modeling, you should examine and clean the data as appropriate.

Your task is to build a binary classifier that predicts `readmitted`. You **must** use a patient-level train/test split so that no patient appears in both the training set and the test set. Use `test_size=0.2` and `random_state=42` for the split.

Write your solution to `/output/model.py`. That file must expose a function with the following exact signature:

```python
def train_and_evaluate(data_path: str) -> dict:
    ...
```

The function should load the data from `data_path`, train the model, and return a dictionary containing at least the key `"auc"` with the ROC-AUC score (as a float) on the held-out test set.

Make sure `/output/model.py` can be imported without side-effects and that `train_and_evaluate` is self-contained (reads data from scratch each call).
