#!/bin/bash
set -u
mkdir -p /logs/verifier
python /tests/verify.py
