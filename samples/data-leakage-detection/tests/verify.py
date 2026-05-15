"""Verifier for data-leakage-detection.

Imports the agent's /output/pipeline_fixed.py, runs train_and_evaluate, and
checks the test accuracy is in the plausible leak-free band [0.70, 0.85].

Also performs source-level static checks: the fixed script must not invoke
StandardScaler / TargetEncoder / mutual_info_classif on the FULL dataset
before the train/test split. We do this by importing AST and looking for the
buggy patterns.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path


REWARD_PATH = Path("/logs/verifier/reward.txt")
FIXED_PATH = Path("/output/pipeline_fixed.py")


def write_reward(value: int, reason: str = "") -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(str(value))
    print(f"reward={value} reason={reason}", file=sys.stderr)


def static_leakage_check(source: str) -> str | None:
    """Return an error string if obvious leakage patterns are detected,
    otherwise None.

    We parse the AST of the agent's `train_and_evaluate` function and look for
    a `train_test_split(...)` call. Any preprocessor that touches data BEFORE
    that split (and that depends on labels or on the full X) is a likely leak.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"

    # Find the train_and_evaluate function body.
    fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "train_and_evaluate":
            fn = node
            break
    if fn is None:
        return "missing function train_and_evaluate"

    # Walk the body and find the index of the first train_test_split call.
    split_idx = None
    for i, stmt in enumerate(fn.body):
        for inner in ast.walk(stmt):
            if isinstance(inner, ast.Call):
                func_name = _get_attr_name(inner.func)
                if func_name == "train_test_split":
                    split_idx = i
                    break
        if split_idx is not None:
            break

    if split_idx is None:
        return "fixed pipeline does not call train_test_split"

    pre_split = fn.body[:split_idx]
    forbidden_calls = {
        "fit_transform",  # StandardScaler / encoder on full data
        "mutual_info_classif",
        "mutual_info_regression",
        "SelectKBest",
    }

    for stmt in pre_split:
        for inner in ast.walk(stmt):
            if isinstance(inner, ast.Call):
                name = _get_attr_name(inner.func)
                if name in forbidden_calls:
                    return f"pre-split call to {name!r} indicates leakage"
                # Detect groupby(...).apply / .mean used to build target encoding
                # on the full dataframe before the split.
                if name in {"mean", "agg", "apply"} and _uses_groupby(inner):
                    return "pre-split target encoding via groupby/mean indicates leakage"

    return None


def _get_attr_name(func: ast.AST) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _uses_groupby(call: ast.Call) -> bool:
    cur = call.func
    while isinstance(cur, ast.Attribute):
        if isinstance(cur.value, ast.Call) and _get_attr_name(cur.value.func) == "groupby":
            return True
        cur = cur.value
    return False


def main() -> int:
    if not FIXED_PATH.exists():
        write_reward(0, "missing /output/pipeline_fixed.py")
        return 0

    source = FIXED_PATH.read_text()
    err = static_leakage_check(source)
    if err is not None:
        write_reward(0, f"static check failed: {err}")
        return 0

    # Import the module and run train_and_evaluate.
    spec = importlib.util.spec_from_file_location("pipeline_fixed", FIXED_PATH)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        write_reward(0, f"failed to import fixed pipeline: {exc}")
        return 0

    if not hasattr(module, "train_and_evaluate"):
        write_reward(0, "fixed pipeline is missing train_and_evaluate")
        return 0

    try:
        acc = module.train_and_evaluate("/workspace/data.csv")
    except Exception as exc:
        write_reward(0, f"train_and_evaluate raised: {exc}")
        return 0

    try:
        acc = float(acc)
    except (TypeError, ValueError):
        write_reward(0, f"return value is not a float: {acc!r}")
        return 0

    if not (0.70 <= acc <= 0.85):
        write_reward(0, f"test accuracy {acc:.4f} outside [0.70, 0.85] band")
        return 0

    write_reward(1, f"all checks passed; accuracy={acc:.4f}")
    return 1


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
