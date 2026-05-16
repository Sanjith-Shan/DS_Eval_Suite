#!/bin/bash
# Run gemini-3-flash-preview against every P1 task with 3 trials each.
# Modelled on /_build/run_gemini_battery.sh.
#
# Requires harbor on PATH, Docker running, and GEMINI_API_KEY exported (or
# present in the repo-root .gemini_key file).

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
JOBS_DIR="${ROOT}/jobs"

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    if [[ -f "${REPO_ROOT}/.gemini_key" ]]; then
        export GEMINI_API_KEY="$(cat "${REPO_ROOT}/.gemini_key")"
    else
        echo "GEMINI_API_KEY not set and ${REPO_ROOT}/.gemini_key missing" >&2
        exit 2
    fi
fi

TASKS=(
    mnar-imputation-destroys-signal
    onehot-rare-categories-overfit
    outlier-removal-kills-minority-class
    normalization-destroys-temporal-feature
    multicollinearity-after-log-transform
    deduplication-loses-valid-longitudinal-data
)

# --ae passes GEMINI_API_KEY into the agent container.
# -k 3 runs three attempts per trial.
# -n 1 single-trial concurrency to stay within 8 GB Docker memory.
for t in "${TASKS[@]}"; do
    job_name="${t}-gemini"
    echo "==> $t (3 trials)"
    if harbor run \
        -p "${ROOT}/tasks/${t}" \
        -a gemini-cli \
        -m google/gemini-3-flash-preview \
        --ae "GEMINI_API_KEY=${GEMINI_API_KEY}" \
        -k 3 \
        -n 1 \
        -o "${JOBS_DIR}" \
        --job-name "${job_name}" \
        -y 2>&1 | tail -25
    then
        echo "  OK"
    else
        echo "  FAILED (continuing)"
    fi
done

echo ""
echo "All tasks done. Summaries:"
for t in "${TASKS[@]}"; do
    rj="${JOBS_DIR}/${t}-gemini/result.json"
    if [[ -f "$rj" ]]; then
        echo "$t:"
        python3 - <<PY
import json
with open("$rj") as f:
    d = json.load(f)
evals = d.get("stats", {}).get("evals", {})
for k, v in evals.items():
    rs = v.get("reward_stats", {}).get("reward", {})
    n = v.get("n_trials", 0)
    n_pass = len(rs.get("1.0", []))
    n_fail = len(rs.get("0.0", []))
    n_err = v.get("n_errors", 0)
    print(f"  {k}: pass={n_pass}/{n} fail={n_fail} err={n_err}")
PY
    else
        echo "$t: NO RESULT"
    fi
done
