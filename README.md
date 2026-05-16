# DS Eval Suite

Harbor-format evaluations of frontier-model data-science ability. The repo
captures three successive iterations of task design (V1 → V2 → V3), each
targeting a different theory of where frontier models fail at data work.

- **V1** — seven thematically-chosen tasks (causal inference, stats, ML,
  ETL). The original deliverable.
- **V2** — tasks reorganised by *failure pattern* rather than topic. Four
  patterns (P1–P4) × ~5–6 tasks each.
- **V3** — context and reference material for the next iteration
  (no tasks committed yet).

Every task is Harbor-format: a sandboxed `environment/`, a known-good
`solution/`, and a programmatic `tests/verify.py`. No LLM-as-judge — every
verifier checks substantive correctness on machine-readable output.

---

## Top-level layout

```
DS_Eval_Suite/
├── README.md                       # this file
├── .gitignore
├── abundant-takehome-context.md    # assignment brief (gitignored)
├── Abundant Research Take Home.pdf # assignment PDF (gitignored)
├── .gemini_key                     # API key (gitignored)
├── .venv/                          # local virtualenv (gitignored)
└── experiments/
    ├── V1/   # original 7-task suite + Gemini eval pipeline
    ├── V2/   # failure-pattern reorganisation (P1–P4)
    └── V3/   # next-iteration context (planning stage)
```

---

## V1 — original 7-task suite

```
experiments/V1/
├── samples/                                # the Harbor tasks (deliverable)
│   ├── confounder-identification/
│   │   ├── instruction.md
│   │   ├── task.toml
│   │   ├── environment/
│   │   │   ├── Dockerfile
│   │   │   └── data.csv
│   │   ├── solution/
│   │   │   ├── solve.py
│   │   │   └── solve.sh
│   │   └── tests/
│   │       ├── test.sh
│   │       └── verify.py
│   ├── ab-test-early-stopping/             # same 4-subdir structure
│   ├── data-leakage-detection/
│   ├── statistical-test-assumptions/
│   ├── etl-timezone-schema-merge/
│   ├── time-series-regime-change/
│   └── simpsons-paradox/
│
├── _build/                                 # data generators + eval pipeline
│   ├── requirements.txt
│   ├── generate_confounder.py
│   ├── generate_ab_test.py
│   ├── generate_leakage.py
│   ├── generate_stats_assumptions.py
│   ├── generate_etl.py
│   ├── generate_timeseries.py
│   ├── generate_simpsons.py
│   ├── smoke_test.py                       # oracle/nop sanity check
│   ├── run_gemini_battery.sh               # 7 tasks × 3 trials → jobs/
│   ├── finalize_logs.sh                    # jobs/ → logs/ reorg
│   ├── make_plots.py                       # logs/ → figures/
│   ├── failure_analysis.py
│   ├── fill_report.py
│   └── build_submission.sh
│
├── logs/                                   # finalised run logs, 3 trials/task
│   ├── confounder-identification/{trial1,trial2,trial3}/
│   ├── ab-test-early-stopping/{trial1,trial2,trial3}/
│   ├── data-leakage-detection/{trial1,trial2,trial3}/
│   ├── statistical-test-assumptions/{trial1,trial2,trial3}/
│   ├── etl-timezone-schema-merge/{trial1,trial2,trial3}/
│   ├── time-series-regime-change/{trial1,trial2,trial3}/
│   └── simpsons-paradox/{trial1,trial2,trial3}/
│
├── jobs/                                   # raw harbor run output (gitignored)
│   └── <task>-gemini/<task>__<runid>/{agent,artifacts,verifier}/
│
└── figures/
    ├── pass_rates.png
    ├── difficulty_curve.png
    └── pass_table.md
```

---

## V2 — failure-pattern reorganisation

Four patterns (P1–P4), each its own self-contained subproject with its own
context doc, tasks, and run logs.

```
experiments/V2/
├── P1_surface-consequence/                 # one fix → one obvious metric change
│   ├── pattern-1-CONTEXT.md
│   ├── GENERAL_REFERENCE.md
│   ├── harbor-reference.md
│   ├── tasks/
│   │   ├── onehot-rare-categories-overfit/{environment,solution,tests}/
│   │   ├── outlier-removal-kills-minority-class/
│   │   ├── normalization-destroys-temporal-feature/
│   │   ├── mnar-imputation-destroys-signal/
│   │   ├── multicollinearity-after-log-transform/
│   │   └── deduplication-loses-valid-longitudinal-data/
│   ├── _build/                             # data gens + "named fix only" probes
│   │   ├── generate_onehot.py / named_fix_only_onehot.py
│   │   ├── generate_outlier.py / named_fix_only_outlier.py
│   │   ├── generate_temporal.py / named_fix_only_temporal.py
│   │   ├── generate_mnar.py / named_fix_only_mnar.py
│   │   ├── generate_multicol.py / named_fix_only_multicol.py
│   │   ├── generate_dedup.py / named_fix_only_dedup.py
│   │   ├── check_no_delta.py
│   │   └── run_gemini_p1.sh
│   ├── results/
│   │   ├── P1_live_eval_report.md
│   │   ├── P1_verification_report.md
│   │   └── gemini_run.log
│   └── jobs/                               # gemini + t1..t6 nop/oracle runs
│
├── P2_cascading-multistep/                 # early bug poisons a downstream step
│   ├── pattern-2-CONTEXT.md
│   ├── P2_report.md
│   ├── tasks/
│   │   ├── wrong-aggregation-cascades-to-wrong-trend/
│   │   ├── wrong-date-parsing-cascades-to-wrong-seasonality/
│   │   ├── wrong-encoding-cascades-to-wrong-model/
│   │   ├── wrong-join-cascades-to-wrong-report/
│   │   └── wrong-sampling-cascades-to-wrong-test/
│   │       (each: generate_data.py + instruction.md + task.toml
│   │              + environment/{data,Dockerfile}
│   │              + solution/{solve.py,solve.sh}
│   │              + tests/{test.sh,verify.py})
│   └── jobs/                               # gemini / nop / oracle per task
│
├── P3_implicit-constraints/                # unstated constraint must be inferred
│   ├── pattern-3-CONTEXT.md
│   ├── P3_REPORT.md
│   ├── tasks/
│   │   ├── p3-class-imbalance-not-mentioned/
│   │   ├── p3-survivorship-bias-in-dataset/
│   │   ├── p3-target-leakage-from-column-name/
│   │   ├── p3-temporal-leakage-from-random-split/
│   │   └── p3-units-mismatch-across-columns/
│   ├── results/                            # (currently empty)
│   └── jobs/                               # gemini-v2 + nop/oracle (both runs)
│
└── P4_overconfidence/                      # claims that outrun the evidence
    ├── pattern-4-CONTEXT.md
    ├── P4_REPORT.md
    ├── tasks/
    │   ├── underpowered-ab-test/
    │   ├── small-sample-strong-claim/
    │   ├── extrapolation-beyond-training/
    │   ├── observational-causal-claim/
    │   └── contradictory-data-sources/
    └── test_results/
        └── jobs/                           # gemini / nop / oracle per task
```

Per-pattern, each task follows the same Harbor layout as V1:

```
tasks/<task-name>/
├── instruction.md          # prompt shown to the agent
├── task.toml               # Harbor task config
├── generate_data.py        # (V2 only) deterministic data generator
├── environment/
│   ├── Dockerfile
│   └── *.csv               # data the agent sees
├── solution/
│   ├── solve.py            # reference oracle solution
│   └── solve.sh
└── tests/
    ├── test.sh
    └── verify.py           # programmatic correctness check
```

Each P*_* folder also keeps copies of the brief PDF and the
`GENERAL_REFERENCE.md` / `harbor-reference.md` so it stays standalone.

---

## V3 — next iteration (context only)

```
experiments/V3/
├── Abundant Research Take Home.pdf
├── CONTEXT.md                  # working notes for V3 design
└── harbor-reference.md
```

No tasks committed yet — V3 is currently in the planning stage.

---

## Prerequisites

```bash
# Docker Desktop must be running
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install harbor
```

Python deps for data regeneration: `pip install -r experiments/V1/_build/requirements.txt`.

---

## Common workflows

### Sanity-check any task

```bash
# Oracle should pass (reward 1), nop should fail (reward 0)
harbor run -p experiments/V1/samples/confounder-identification -a oracle -y
harbor run -p experiments/V1/samples/confounder-identification -a nop    -y

# Same shape for V2 tasks
harbor run -p experiments/V2/P1_surface-consequence/tasks/onehot-rare-categories-overfit -a oracle -y
```

### Run the V1 Gemini battery (7 × 3)

```bash
export GEMINI_API_KEY=<your key>
bash experiments/V1/_build/run_gemini_battery.sh    # → jobs/
bash experiments/V1/_build/finalize_logs.sh         # jobs/ → logs/
.venv/bin/python experiments/V1/_build/make_plots.py   # → figures/
```

### Run a V2 pattern battery

```bash
export GEMINI_API_KEY=<your key>
bash experiments/V2/P1_surface-consequence/_build/run_gemini_p1.sh
# results land under experiments/V2/P1_surface-consequence/jobs/
```

### Regenerate a task's bundled data

```bash
# V1
.venv/bin/python experiments/V1/_build/generate_confounder.py

# V2 (per-task generator)
.venv/bin/python experiments/V2/P1_surface-consequence/_build/generate_onehot.py
.venv/bin/python experiments/V2/P2_cascading-multistep/tasks/wrong-aggregation-cascades-to-wrong-trend/generate_data.py
```

All generators use fixed seeds — output is reproducible.

---

## Notes

- `tests/` is mounted only at verifier time; ground truth never lives in
  `environment/`.
- `jobs/` is gitignored; finalised, named runs live under `logs/` (V1) or
  `<pattern>/jobs/` and `<pattern>/results/` (V2).
- Each V2 pattern is self-contained — its CONTEXT.md, report, tasks, and
  runs can be read without the others.
- See `experiments/V1/_build/fill_report.py` and the per-pattern
  `P*_REPORT.md` files for the methodology, design rationale, and measured
  pass@1/pass@3 against `gemini-3-flash-preview`.
