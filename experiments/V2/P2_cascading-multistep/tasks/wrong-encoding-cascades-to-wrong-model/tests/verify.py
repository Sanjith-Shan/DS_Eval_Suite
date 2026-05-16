"""Verifier for wrong-encoding-cascades-to-wrong-model.

Reward 1 iff:
  - /output/results.json exists and is valid JSON
  - Required keys: accuracy, top_features, model_type
  - accuracy >= 0.79
  - top_features is a list of exactly 5 strings
  - top_features contains "education_level" (exact, case-insensitive) as a bare
    feature name — i.e. the ordinal encoding was used, not one-hot
  - top_features does NOT contain one-hot fragment names like "education_level_phd",
    "education_level_masters", "education_level_bachelors", "education_level_high_school"

Cascade: an agent that defaults to pd.get_dummies() or OneHotEncoder for all
categoricals will report fragmented dummy names (education_level_phd, etc.) in
top_features. The correct approach is to recognise the ordinal nature of
education_level and encode it as a single integer (0–3), which then appears as
the dominant top feature under its original column name "education_level".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REWARD_PATH = Path("/logs/verifier/reward.txt")
OUTPUT_PATH = Path("/output/results.json")

# One-hot fragment patterns that indicate wrong encoding
ONE_HOT_FRAGMENTS = {
    "education_level_phd",
    "education_level_masters",
    "education_level_bachelors",
    "education_level_high_school",
    # also catch sklearn-style name__value patterns
    "education_level__phd",
    "education_level__masters",
    "education_level__bachelors",
    "education_level__high_school",
    # and cat__education_level_* patterns
}


def write_reward(value: int, reason: str = "") -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(str(value))
    print(f"reward={value} reason={reason}", file=sys.stderr)


def main() -> int:
    if not OUTPUT_PATH.exists():
        write_reward(0, "missing /output/results.json")
        return 0

    try:
        data = json.loads(OUTPUT_PATH.read_text())
    except json.JSONDecodeError as exc:
        write_reward(0, f"invalid JSON: {exc}")
        return 0

    required = {"accuracy", "top_features", "model_type"}
    missing = required - data.keys()
    if missing:
        write_reward(0, f"missing keys: {sorted(missing)}")
        return 0

    # Accuracy check
    try:
        accuracy = float(data["accuracy"])
    except (TypeError, ValueError):
        write_reward(0, "accuracy must be numeric")
        return 0

    if accuracy < 0.79:
        write_reward(
            0,
            f"accuracy {accuracy:.4f} < 0.79; likely caused by wrong encoding "
            f"or missing a strong predictor",
        )
        return 0

    # top_features check
    top_features = data["top_features"]
    if not isinstance(top_features, list):
        write_reward(0, "top_features must be a list")
        return 0

    if len(top_features) != 5:
        write_reward(0, f"top_features must have exactly 5 entries, got {len(top_features)}")
        return 0

    if not all(isinstance(f, str) for f in top_features):
        write_reward(0, "all entries in top_features must be strings")
        return 0

    top_lower = [f.strip().lower() for f in top_features]

    # Check for one-hot fragments — these indicate wrong encoding
    for fragment in ONE_HOT_FRAGMENTS:
        if fragment in top_lower:
            write_reward(
                0,
                f"top_features contains '{fragment}' — this is a one-hot dummy name. "
                f"education_level has a natural ordinal ordering; encoding it as a "
                f"single integer preserves the monotonic signal and causes it to appear "
                f"as 'education_level' (not a fragmented dummy) in feature importance. "
                f"top_features={top_features}",
            )
            return 0

    # Check that the bare "education_level" feature appears
    if "education_level" not in top_lower:
        write_reward(
            0,
            f"top_features does not contain 'education_level'. "
            f"With correct ordinal encoding, education_level should be the top-ranked "
            f"feature. Got top_features={top_features}",
        )
        return 0

    # model_type just needs to be a non-empty string
    model_type = str(data["model_type"]).strip()
    if not model_type:
        write_reward(0, "model_type must be a non-empty string")
        return 0

    write_reward(
        1,
        f"all checks passed: accuracy={accuracy:.4f}, "
        f"education_level in top_features, model_type={model_type!r}",
    )
    return 1


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
