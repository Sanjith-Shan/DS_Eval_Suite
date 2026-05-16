"""
Verifier for p3-class-imbalance-not-mentioned.

Reads /output/predictions.csv (agent output) and /tests/holdout_labels.csv (hidden ground truth).
Computes F1 (pos_label=1). Reward=1 if F1 > 0.15 AND predictions are not constant.
"""

import os
import sys
import pandas as pd
from sklearn.metrics import f1_score

PREDICTIONS_PATH  = "/output/predictions.csv"
LABELS_PATH       = "/tests/holdout_labels.csv"
REWARD_PATH       = "/logs/verifier/reward.txt"
LOG_PATH          = "/logs/verifier/output.log"


def write_log(reward: float, reason: str):
    os.makedirs(os.path.dirname(REWARD_PATH), exist_ok=True)
    with open(REWARD_PATH, "w") as f:
        f.write(str(int(reward)))
    with open(LOG_PATH, "w") as f:
        f.write(f"reward={reward}\nreason={reason}\n")
    print(f"reward={reward}  reason={reason}")


def main():
    # 1. Check predictions file exists
    if not os.path.isfile(PREDICTIONS_PATH):
        write_log(0.0, f"Missing {PREDICTIONS_PATH}")
        sys.exit(0)

    # 2. Load predictions
    try:
        preds = pd.read_csv(PREDICTIONS_PATH)
    except Exception as e:
        write_log(0.0, f"Could not read {PREDICTIONS_PATH}: {e}")
        sys.exit(0)

    # 3. Validate required columns
    required_cols = {"id", "predicted_class"}
    missing_cols = required_cols - set(preds.columns)
    if missing_cols:
        write_log(0.0, f"predictions.csv missing columns: {sorted(missing_cols)}")
        sys.exit(0)

    # 4. Validate predicted_class values are in {0, 1}
    invalid = preds[~preds["predicted_class"].isin([0, 1])]
    if len(invalid) > 0:
        write_log(0.0, f"predicted_class has values outside {{0,1}}: {invalid['predicted_class'].unique()[:5].tolist()}")
        sys.exit(0)

    # 5. Load ground truth labels
    if not os.path.isfile(LABELS_PATH):
        write_log(0.0, f"Missing ground truth {LABELS_PATH} — verifier misconfiguration")
        sys.exit(0)

    try:
        labels = pd.read_csv(LABELS_PATH)
    except Exception as e:
        write_log(0.0, f"Could not read {LABELS_PATH}: {e}")
        sys.exit(0)

    # 6. Join on id — every holdout id must be present in predictions
    merged = labels.merge(preds[["id", "predicted_class"]], on="id", how="left")
    n_missing = merged["predicted_class"].isna().sum()
    if n_missing > 0:
        write_log(0.0, f"{n_missing} holdout IDs missing from predictions.csv")
        sys.exit(0)

    merged["predicted_class"] = merged["predicted_class"].astype(int)

    # 7. Reject constant predictions (all-zeros or all-ones)
    unique_preds = merged["predicted_class"].unique()
    if len(unique_preds) == 1:
        val = unique_preds[0]
        write_log(
            0.0,
            f"predicted_class is constant (all {val}) — model predicts only one class. "
            "This indicates the class imbalance was not handled. F1=0."
        )
        sys.exit(0)

    # 8. Compute F1
    y_true = merged["target"].values
    y_pred = merged["predicted_class"].values
    f1 = f1_score(y_true, y_pred, pos_label=1)

    # 9. Pass/fail threshold
    F1_THRESHOLD = 0.15
    if f1 > F1_THRESHOLD:
        write_log(1.0, f"F1={f1:.4f} > {F1_THRESHOLD} — passed. Model handled class imbalance.")
    else:
        write_log(
            0.0,
            f"F1={f1:.4f} <= {F1_THRESHOLD} — failed. "
            "Model likely did not handle the 97/3 class imbalance correctly."
        )


if __name__ == "__main__":
    main()
