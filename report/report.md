# DS Eval Suite — Report

**Author:** Sanjith Shanmugavel
**Model under test:** `google/gemini-3-flash-preview` (via Harbor `gemini-cli` agent)
**Submission date:** 2026-05-15
**Repository:** https://github.com/Sanjith-Shan/DS_Eval_Suite

---

## 1. Distribution — Why this slice of data science?

I deliberately scoped this eval narrowly: **seven tasks that test the judgement
calls a senior DS makes in their first hour with a new dataset** — not the
coding mechanics of producing a plot or fitting a model. Every task targets a
moment where the *correct procedural action* is one most humans agree on but
most frontier models still skip.

### Slice

| Failure category | Task |
|---|---|
| Causal vs. correlational interpretation | `confounder-identification`, `simpsons-paradox` |
| Experimentation discipline | `ab-test-early-stopping` |
| Statistical methodology under realistic violations | `statistical-test-assumptions` |
| ML pipeline correctness (leakage debugging) | `data-leakage-detection` |
| Time-series structural change | `time-series-regime-change` |
| Production ETL with schema/timezone drift | `etl-timezone-schema-merge` |

These categories trace directly to documented failure modes in the recent
benchmark literature (see §3). They share four properties:

1. **The wrong answer is the easy answer.** A model that follows the surface
   prompt — "is treatment B better?", "does X predict Y?", "is this
   significant?" — gets it wrong. The right answer requires reading something
   the prompt didn't explicitly highlight (the test plan, the README, the
   structural break, the confounder).
2. **The error is *behavioural*, not knowledge.** A model that has read about
   Simpson's paradox can still call Treatment B better in aggregate. We are
   measuring whether it will *do the stratification* before answering.
3. **Verifiers are deterministic.** Every reward in this suite is a binary,
   content-aware check on machine-readable output (JSON, CSV, or a Python
   script). No LLM-as-judge; no fragile keyword regex; no test-set leakage.
4. **Solutions are real.** Every task ships with a working reference solution
   that passes the verifier. I ran each through `harbor run -a oracle` and
   `harbor run -a nop` to confirm `oracle=1, nop=0` before evaluating Gemini.

### What's deliberately out of scope

- **Pure code generation** (writing a model from scratch, implementing a
  paper). Covered by HumanEval / MLE-bench / SWE-bench; not a DS-specific
  failure mode.
- **Long-horizon multi-day projects.** A take-home this size can't isolate
  signal in 24-hour traces.
- **Multimodal inputs** (charts, screenshots, PDFs). Powerful angle, but
  including it would mix vision failures with reasoning failures. I'd add
  these in the scale-up (§4).
- **LLM-as-judge tasks** ("write a memo for the VP"). Reward becomes a
  judgment call; the noise floor swamps signal at n=3.
- **Web-browsing / agentic research.** Not part of the canonical DS workflow
  I'm targeting.

### How I'd narrow further if forced

If I had to pick three tasks that I'm most confident isolate genuine difficulty
(not just verifier strictness), they'd be: `data-leakage-detection`,
`statistical-test-assumptions`, and `etl-timezone-schema-merge`. They require
the model to discover a problem the prompt only hints at, then act on that
discovery — the failure modes the brief explicitly cared about.

---

## 2. Difficulty profile — pass@1 / pass@3 against `gemini-3-flash-preview`

Each task was run with `harbor run -a gemini-cli -m google/gemini-3-flash-preview -k 3`
(three independent attempts per trial) on a single MacBook Pro (M-series,
8 GB Docker quota). Raw transcripts are in `logs/` (one ATIF JSON per attempt).

### Aggregate

> **(Filled in once the live battery completes — see `report/figures/pass_table.md`.)**

```
{{PASS_TABLE}}
```

### Per-task bar chart

![pass rates](figures/pass_rates.png)

### Difficulty curve (sorted ascending)

![difficulty curve](figures/difficulty_curve.png)

### What dominates the failure mode

A short, qualitative read of the trajectories (full failure-mode breakdown
appears in §5):

- The model frequently gives the answer the *first sentence* of the prompt
  invites, missing the second-order constraint.
- When verifiers ask for a JSON schema, the model occasionally produces
  near-passing output but with one field type wrong (string instead of bool,
  missing key) — a real DS-collaboration failure, not a verifier bug.
- For the ETL and time-series tasks, code-generation is fine but the model
  *picks the wrong default*: silent `nonexistent='raise'` on DST timestamps,
  full-history mean for forecasting.

---

## 3. Research awareness

The seven tasks were built by mining 2024–2026 DS benchmark literature for
failure modes that (a) every published frontier model still misses, (b) are
*caused* by judgement gaps rather than missing capabilities, and (c) can be
isolated in a self-contained Harbor task.

| Source | What I borrowed |
|---|---|
| **BLADE: Benchmarking Language model Agents for Data-driven Analysis** (2025) | Coverage gap on statistical methodology (<13% of expert decisions); inspired `statistical-test-assumptions`. |
| **CausalPitfalls** (2025) | Every frontier model defaults to causal language under strong correlation; inspired `confounder-identification`. |
| **DABstep** (2024) | 16% aggregate pass on production analytics steps; inspired the format of `ab-test-early-stopping`. |
| **QRData** (2024) | Models score worse on subgroup-stratification questions; inspired `simpsons-paradox`. |
| **MLE-bench** (2024) | Agents struggle to debug and recover; inspired `data-leakage-detection` with three layered bugs. |
| **KramaBench** (2024) | Only benchmark testing real ETL; inspired `etl-timezone-schema-merge` (DST + schema drift). |
| **Gao et al., LLMs and Time Series** (2025) | ~37% of LLM time-series accuracy is memorisation; inspired `time-series-regime-change`. |
| **DSAEval** (referenced in brief) | Format pattern for self-contained Harbor tasks. |

### Production tools / patterns referenced

- **scikit-learn `Pipeline`** as the canonical leak-free preprocessing pattern.
- **pandas `tz_localize(nonexistent=...)`** docs as the canonical DST gotcha.
- **Causal Inference: The Mixtape** (Cunningham) for the partial-correlation
  framing in `confounder-identification`.
- **Kohavi et al., "Trustworthy Online Controlled Experiments"** for the
  sequential-testing setup in `ab-test-early-stopping`.

### Loops I automated

Per the brief's prompt to lean on coding agents — I built this with Claude
Code. Specifically:

- **Data generators.** Seed-search loops where I needed specific empirical
  properties (e.g. the A/B-test data had to satisfy `p_day5 < 0.05 AND p_day14 > 0.2`).
  Claude wrote the search-over-seeds script; I picked the seed.
- **Verifier AST checks.** The leakage verifier inspects the agent's fixed
  pipeline AST to confirm `fit_transform` / `SelectKBest` are not called pre-
  `train_test_split`. Claude drafted the AST walker.
- **Local smoke harness** (`_build/smoke_test.py`) — a path-patching shim that
  let me iterate quickly on each task before installing Harbor.

What I *did* hand-tune: the verifier tolerance bands, the JSON schemas in
each task's instructions, and the choice of distribution / parameters in
each data generator. Those are the parts that decide whether the task
measures what I claim it measures.

---

## 4. Scale plan — 10 → 1,000 tasks

Going from 10 to 1,000 hand-built tasks doesn't scale linearly. The plan I'd
defend in interview:

### Mining (450 tasks)

- **Public Kaggle notebooks.** Filter for notebooks tagged
  `eda`, `feature-engineering`, `time-series`, `experimentation`. Each one
  yields 1–3 "moment of judgement" tasks: pick a notebook, automate the data
  fetch into the environment, write a verifier that scores the *decision the
  author made*. Aim for 300 tasks via this route.
- **DABstep / BLADE / QRData / KramaBench reproduction.** Re-implement
  failing items from these benchmarks as Harbor tasks with stricter
  verifiers. ~100 tasks.
- **Production-style ETL exemplars.** Pull from dbt / Airflow / dagster
  example projects; plant realistic bugs (timezone, schema, deduplication,
  late-arriving data). ~50 tasks.

### Templated synthesis (400 tasks)

Each of the 7 task families in this submission is a *template* parameterised
by:
- distribution parameters (e.g. lognormal σ for stats-assumptions),
- magnitude of the effect (regime shift size, leakage strength),
- presence/absence of a "red herring" feature in the instruction.

For the confounder family alone, a parameterised generator yields O(50)
distinct variants (different domains: ice-cream/drowning, advertising/sales,
class size/grades, etc.) with verifiers that follow the same JSON schema.
Total: ~50 variants × 7 families ≈ **350–400 synthetic tasks**, each one
fresh enough to not be memorisable.

### Frontier-failure mining (150 tasks)

Run a strong-but-imperfect model (Gemini 3 Pro, Claude Opus 4.7) against
unlabelled DS notebooks, ask for a critique of each, then *invert*: turn
each critique into a verifier that scores whether the model would catch it.
Hard, but it surfaces failures the literature hasn't named yet.

### QA loop

For every minted task:

1. **Oracle check.** `harbor run -a oracle` returns reward 1 — proves
   solvability.
2. **Nop check.** `harbor run -a nop` returns reward 0 — proves the verifier
   isn't auto-passing.
3. **Frontier-pass check.** If reward 1 on *both* oracle and a frontier
   model with 3 trials, the task has no headroom — drop it.
4. **Frontier-fail check.** If reward 0 on frontier × 3 trials *and* the
   trajectory shows a real failure (not "model crashed compiling the
   import"), the task is keep-worthy.
5. **`harbor check` against TB3 rubric** to catch authoring issues.
6. **Per-family balance.** Aim for ~150 tasks per family, with a
   difficulty range from 10% to 60% pass@3.

### What I'd *not* spend on

- More leaderboards. The marginal task in slot 1,001 doesn't move pass@3.
- Heavier verifiers. LLM-as-judge inflates noise without adding signal at
  this scale.
- New environments. Docker + Python is the universal denominator; adding
  R / Julia / Spark increases maintenance burden without improving
  difficulty discovery.

---

## 5. Failure analysis (from the Gemini trajectories)

> **(Filled in once the live battery completes — pulls a representative
> failed trajectory per task and diagnoses why it failed, distinguishing
> genuine task difficulty from task-design bugs.)**

```
{{FAILURE_ANALYSIS}}
```

---

## Appendix — running the suite

```bash
# Prereqs
brew install docker  # or Docker Desktop
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install harbor
docker info >/dev/null   # confirm daemon

# Sanity (oracle should pass, nop should fail)
for t in samples/*/; do
  harbor run -p "$t" -a oracle -y -o jobs --job-name "$(basename "$t")-oracle"
  harbor run -p "$t" -a nop    -y -o jobs --job-name "$(basename "$t")-nop"
done

# Full Gemini battery (3 trials per task)
export GEMINI_API_KEY=<your key>
bash _build/run_gemini_battery.sh

# Plots and tables
.venv/bin/python _build/make_plots.py
```

### Known limitations

- I could not run `harbor check` against the TB3 rubric — that command is
  hardwired to Anthropic Sonnet as the judge model and I only have a Gemini
  key. I documented the rubric checks I'd care about (see §4 QA loop) and
  validated each task by hand. Concretely: instructions are 1–2 paragraphs
  and don't leak the answer; verifiers parse output and check substantive
  correctness rather than keyword presence; no test-set artifacts in
  `environment/`; every task has a working `solve.py` reference.
- Data generators use fixed seeds — task data is reproducible but not novel
  per trial. In the scale-up (§4) every parameterised variant gets a fresh
  seed.
