#!/bin/bash
# Reorganise harbor's jobs/ output into the submission's logs/ layout.
#   jobs/<task>-gemini/<trial_uid>/{agent,verifier,...}
# becomes
#   logs/<task>/trial<N>/{agent,verifier,...}
#
# Also writes logs/summary.json with per-task / overall pass@1, pass@3.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOBS="${ROOT}/jobs"
LOGS="${ROOT}/logs"

# Wipe stale logs but keep README.md.
find "${LOGS}" -mindepth 1 -maxdepth 1 ! -name README.md -exec rm -rf {} +

TASKS=(
    confounder-identification
    ab-test-early-stopping
    data-leakage-detection
    statistical-test-assumptions
    etl-timezone-schema-merge
    time-series-regime-change
    simpsons-paradox
)

for t in "${TASKS[@]}"; do
    src="${JOBS}/${t}-gemini"
    dst="${LOGS}/${t}"
    [[ -d "$src" ]] || { echo "MISSING: $src"; continue; }
    mkdir -p "$dst"
    cp "${src}/result.json" "${dst}/job_result.json" 2>/dev/null || true
    cp "${src}/job.log" "${dst}/job.log" 2>/dev/null || true
    n=0
    for trial in "${src}"/*/; do
        [[ -d "$trial" ]] || continue
        case "$(basename "$trial")" in
            *__*) ;;            # Harbor's trial dirs match this glob
            *) continue ;;
        esac
        n=$((n+1))
        cp -r "$trial" "${dst}/trial${n}"
    done
done

# Generate summary.json
python3 - "${LOGS}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
summary = {}
for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    rewards = []
    for trial in sorted(task_dir.glob("trial*")):
        f = trial / "verifier" / "reward.txt"
        try:
            rewards.append(int(float(f.read_text().strip())))
        except Exception:
            rewards.append(0)
    if rewards:
        p1 = sum(rewards) / len(rewards)
        p3 = int(any(r == 1 for r in rewards[:3]))
        summary[task_dir.name] = {"rewards": rewards, "pass_at_1": p1, "pass_at_3": p3}
overall_p1 = sum(t["pass_at_1"] for t in summary.values()) / max(1, len(summary))
overall_p3 = sum(t["pass_at_3"] for t in summary.values()) / max(1, len(summary))
summary["__aggregate__"] = {"pass_at_1": overall_p1, "pass_at_3": overall_p3}
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
