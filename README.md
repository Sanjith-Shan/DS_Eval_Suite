# DS Eval Suite

Seven Harbor-format tasks evaluating an AI agent's data-science ability. Each task targets a documented frontier-model failure mode in statistical reasoning, causal inference, ML correctness, or data-pipeline debugging.

## Layout

```
samples/                        # Harbor tasks (the deliverable)
├── confounder-identification/
├── ab-test-early-stopping/
├── data-leakage-detection/
├── statistical-test-assumptions/
├── etl-timezone-schema-merge/
├── time-series-regime-change/
└── simpsons-paradox/
report/                         # Methodology, rationale, results
logs/                           # Harbor run output (populated by run_eval.sh)
_build/                         # Data-generation scripts (out-of-band of Harbor)
run_eval.sh                     # Convenience runner for all 7 tasks × 3 trials
```

## Running a task

```bash
# Sanity-check the reference solution
harbor run -p samples/confounder-identification -a oracle

# Sanity-check that inaction fails the verifier
harbor run -p samples/confounder-identification -a nop

# Evaluate the target model
harbor run -p samples/confounder-identification -a gemini-cli -m google/gemini-3-flash-preview
```

Run the whole battery with `./run_eval.sh` after exporting `GEMINI_API_KEY`.

## Reproducing the bundled data

The CSVs under each task's `environment/` are generated deterministically (fixed seeds). To regenerate from scratch:

```bash
python3 -m venv .venv && .venv/bin/pip install -r _build/requirements.txt
.venv/bin/python _build/generate_all.py
```

## Notes

- Every verifier is content-aware: it parses agent output and checks substantive correctness, not surface keywords.
- No reference answers leak into `environment/` — verifiers live under `tests/` which is mounted only at verification time.
- See `report/report.md` for design rationale and the empirical evidence behind each failure mode.
