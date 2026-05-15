#!/bin/bash
# Convenience runner for all 7 tasks against the target model.
# Usage:
#   export GEMINI_API_KEY=<your-key>
#   ./run_eval.sh                 # full battery (3 trials per task)
#   ./run_eval.sh sanity          # oracle + nop sanity passes only
#   ./run_eval.sh <task-name>     # 3 trials on a single task

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOGS="${ROOT}/logs"
SAMPLES="${ROOT}/samples"

if [[ -z "${GEMINI_API_KEY:-}" ]] && [[ "${1:-}" != "sanity" ]]; then
    echo "GEMINI_API_KEY is not set. Export it before running the full battery." >&2
    exit 1
fi

if ! command -v harbor >/dev/null 2>&1; then
    echo "harbor CLI not found in PATH. Install Harbor before running this script." >&2
    exit 1
fi

mkdir -p "${LOGS}"

ALL_TASKS=(
    confounder-identification
    ab-test-early-stopping
    data-leakage-detection
    statistical-test-assumptions
    etl-timezone-schema-merge
    time-series-regime-change
    simpsons-paradox
)

if [[ "${1:-}" == "sanity" ]]; then
    for t in "${ALL_TASKS[@]}"; do
        echo ">>> oracle: ${t}"
        harbor run -p "${SAMPLES}/${t}" -a oracle 2>&1 | tee "${LOGS}/${t}.oracle.log"
        echo ">>> nop:    ${t}"
        harbor run -p "${SAMPLES}/${t}" -a nop 2>&1    | tee "${LOGS}/${t}.nop.log"
    done
    exit 0
fi

if [[ -n "${1:-}" ]]; then
    TASKS=("$1")
else
    TASKS=("${ALL_TASKS[@]}")
fi

for t in "${TASKS[@]}"; do
    for trial in 1 2 3; do
        echo ">>> gemini trial ${trial}: ${t}"
        harbor run -p "${SAMPLES}/${t}" -a gemini-cli -m google/gemini-3-flash-preview 2>&1 \
            | tee "${LOGS}/${t}.gemini.trial${trial}.log"
    done
done

echo "Done. Logs written under ${LOGS}/."
