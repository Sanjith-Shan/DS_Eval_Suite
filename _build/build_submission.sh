#!/bin/bash
# Build the submission zip per the brief: samples/, logs/, report/ only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/submission.zip"

cd "$ROOT"

if [[ ! -d "samples" ]] || [[ ! -d "logs" ]] || [[ ! -d "report" ]]; then
    echo "Missing one of samples/, logs/, report/" >&2
    exit 2
fi

rm -f "$OUT"
zip -r "$OUT" samples logs report \
    -x "*.DS_Store" \
    -x "**/__pycache__/*" \
    -x "*/jobs/*"

echo
echo "Submission zip: $OUT"
echo "Size: $(du -h "$OUT" | cut -f1)"
echo
echo "Top-level contents:"
unzip -l "$OUT" | head -30
echo
echo "Files: $(unzip -l "$OUT" | tail -1 | awk '{print $2}')"
