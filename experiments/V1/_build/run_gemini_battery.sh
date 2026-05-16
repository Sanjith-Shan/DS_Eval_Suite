#!/bin/bash
# Run the gemini-3-flash-preview eval against every task with 3 trials each.
# Requires harbor on PATH, Docker running, and GEMINI_API_KEY exported.
#
# Strategy: invoke harbor with -k 3 (three attempts per trial) for a single
# task at a time so Docker doesn't get oversubscribed. Each task gets its
# own job directory under jobs/.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOBS_DIR="${ROOT}/jobs"

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    if [[ -f "${ROOT}/.gemini_key" ]]; then
        export GEMINI_API_KEY="$(cat "${ROOT}/.gemini_key")"
    else
        echo "GEMINI_API_KEY not set and .gemini_key missing" >&2
        exit 2
    fi
fi

TASKS=(
    confounder-identification
    ab-test-early-stopping
    data-leakage-detection
    statistical-test-assumptions
    etl-timezone-schema-merge
    time-series-regime-change
    simpsons-paradox
)

# --ae passes GEMINI_API_KEY into the agent container.
# -k 3 runs three attempts per trial.
# -n 1 single-trial concurrency to stay within 8 GB Docker memory.
for t in "${TASKS[@]}"; do
    job_name="${t}-gemini"
    echo "==> $t (3 trials)"
    if harbor run \
        -p "${ROOT}/samples/${t}" \
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

echo "All tasks done. Summaries:"
for t in "${TASKS[@]}"; do
    rj="${JOBS_DIR}/${t}-gemini/result.json"
    if [[ -f "$rj" ]]; then
        echo "$t:"
        cat "$rj" | python3 -c "import json,sys; d=json.load(sys.stdin); print(' ', json.dumps({k:d.get(k) for k in ('mean','trials','exceptions') if k in d}))"
    fi
done
