"""
Verifier for p3-target-leakage-from-column-name.

Scoring rules:
  1. /output/results.json must exist.
  2. Required fields: accuracy (float), model_type (str).
  3. accuracy must be in [0.73, 0.84].
     - ~0.98 from keeping leaky columns → fails (too high).
     - Models that don't learn → fail (too low).
     - Only models that dropped the two leaky columns land in [0.73, 0.84].

reward=1 only when all checks pass.
"""

import json
import os
import sys

RESULTS_PATH = "/output/results.json"
REWARD_PATH = "/logs/verifier/reward.txt"
LOG_PATH = "/logs/verifier/output.log"

ACC_LOW = 0.73
ACC_HIGH = 0.84


def _write(reward: int, reason: str):
    os.makedirs(os.path.dirname(REWARD_PATH), exist_ok=True)
    with open(REWARD_PATH, "w") as f:
        f.write(str(reward))
    with open(LOG_PATH, "w") as f:
        f.write(f"reward={reward}\nreason={reason}\n")


def fail(reason: str):
    _write(0, reason)
    print(f"FAIL: {reason}", file=sys.stderr)
    sys.exit(0)


def succeed(reason: str):
    _write(1, reason)
    print(f"PASS: {reason}", file=sys.stderr)
    sys.exit(0)


def main():
    # Step 1: file existence
    if not os.path.isfile(RESULTS_PATH):
        fail(f"Missing {RESULTS_PATH}")

    # Step 2: parse JSON and required fields
    try:
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    except Exception as e:
        fail(f"Could not parse {RESULTS_PATH}: {e}")

    for field in ("accuracy", "model_type"):
        if field not in results:
            fail(f"Missing required field: '{field}'")

    accuracy = results["accuracy"]
    model_type = results["model_type"]

    if not isinstance(accuracy, (int, float)):
        fail("'accuracy' must be a number")
    if not isinstance(model_type, str):
        fail("'model_type' must be a string")

    # Step 3: accuracy band check
    if not (ACC_LOW <= float(accuracy) <= ACC_HIGH):
        fail(
            f"accuracy={accuracy:.4f} is outside the acceptable range "
            f"[{ACC_LOW}, {ACC_HIGH}]. "
            "A value above the range suggests target leakage; "
            "a value below suggests an invalid model."
        )

    succeed(
        f"accuracy={accuracy:.4f} within [{ACC_LOW}, {ACC_HIGH}], "
        f"model_type={model_type!r}."
    )


if __name__ == "__main__":
    main()
