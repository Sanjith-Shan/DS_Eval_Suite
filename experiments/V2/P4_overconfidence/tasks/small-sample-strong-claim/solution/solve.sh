#!/bin/bash
set -eu
mkdir -p /output
DIR="$(cd "$(dirname "$0")" && pwd)"
python "$DIR/solve.py"
