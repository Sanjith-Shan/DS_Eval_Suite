# DS Eval Suite

Seven Harbor-format tasks evaluating an AI agent's data-science ability,
targeting documented frontier-model failure modes in statistical reasoning,
causal inference, ML correctness, and production ETL.

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
logs/                           # `harbor run -a gemini-cli` output, 3 trials/task
_build/                         # Data generators, plot/analysis utilities, runner
```

## Prerequisites

```bash
# Docker Desktop (must be running)
# Then:
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install harbor
```

## Sanity-check the tasks

```bash
# Oracle should pass (reward 1)
harbor run -p samples/confounder-identification -a oracle -y

# Nop should fail (reward 0)
harbor run -p samples/confounder-identification -a nop -y
```

## Run the Gemini eval

```bash
export GEMINI_API_KEY=<your key>
bash _build/run_gemini_battery.sh    # 7 tasks × 3 trials → jobs/
bash _build/finalize_logs.sh         # reorganises jobs/ → logs/
.venv/bin/python _build/make_plots.py   # writes report/figures/
```

## Regenerating the bundled task data

Each task's `environment/` ships pre-generated CSVs (fixed seed, reproducible).
To regenerate:

```bash
python3 -m venv .venv && .venv/bin/pip install -r _build/requirements.txt
for g in _build/generate_*.py; do .venv/bin/python "$g"; done
```

## Notes

- Every verifier checks substantive correctness on machine-readable output
  (JSON / CSV / a runnable Python script). No LLM-as-judge.
- `tests/` is mounted only at verifier time; ground truth never lives in
  `environment/`.
- See `report/report.md` for design rationale, the empirical evidence behind
  each failure mode, measured pass@1/pass@3 against `gemini-3-flash-preview`,
  and a sample failure analysis.
