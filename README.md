# Post-Analysis Eval Suite

An evaluation framework that systematically identifies and tests a blind spot in frontier AI models. Through iterative experimentation across 40+ candidate tasks, I discovered that Gemini Flash consistently skips post-analysis validation steps that any senior data scientist would perform automatically. I then built a focused 10-task suite around that finding, achieving a 0% pass rate across all 30 trials.

## The Problem

Frontier language models can execute data science workflows competently. They fit regressions, run hypothesis tests, build classifiers, and handle common pitfalls like Simpson's paradox or data leakage. But executing an analysis and validating an analysis are different skills. A senior data scientist does not just produce a result. They check whether the method they chose was appropriate for the data they received, whether the assumptions behind the method actually hold, and whether their own actions introduced new problems.

I set out to test whether Gemini Flash performs these validation steps on its own. It does not.

## The Approach

Rather than guessing which tasks would be hard, I ran a structured search.

**Round 1 (7 tasks).** Broad probing across causal inference, A/B testing, Simpson's paradox, ANOVA assumptions, ETL edge cases, time-series analysis, and data leakage. Gemini solved 6 of 7. The one failure revealed the pattern. The model fixed a data leakage bug perfectly but never checked whether the fix introduced multicollinearity. It applied the technique and stopped without looking at its own output.

**Round 2 (21 tasks).** Four hypotheses about failure patterns, each tested with 5-6 tasks. Most hypotheses were wrong. But the tasks that consistently caught the model, regardless of which hypothesis they came from, all shared the same trait. They required the model to do something the instructions never asked for, based purely on what the data demanded.

**Round 3 (10 tasks).** A focused suite built around the validated pattern, targeting **post-analysis validation** as a coherent slice of data science. Every task asks for a standard analysis where the data violates an assumption the instruction never mentions. The model must independently discover the violation and adjust.

## Results

All 10 tasks achieve 0% pass@3 against gemini-3-flash-preview (0 passes in 30 trials).

| Task | What the Model Misses | Gap Between Naive and Correct |
|------|----------------------|-------------------------------|
| Multicollinearity after log transform | VIF check after variance-stabilizing transform | p = 0.12 vs p = 0.03 |
| Clustered parametric test | Independence assumption for t-test on clustered data | Wrong test type entirely |
| Survivorship bias in sample | Selection bias from non-representative sampling | Missing bias acknowledgment |
| Autocorrelated residuals | Durbin-Watson check on time-ordered residuals | p = 0.026 vs p = 0.070 |
| Multiple comparisons | Bonferroni/BH correction across 28 pairwise tests | 7 pairs vs 2 pairs |
| Clustered treatment assignment | Pseudo-replication from group-level treatment | p = 0.000014 vs p = 0.131 |
| Censored survival data | Right-censoring requiring survival analysis | p = 0.013 vs p = 0.790 |
| Spurious regression | Stationarity check on trending time series | R² = 0.86 vs R² = 0.015 |
| Regression to the mean | Using comparison group for selected subgroup | p < 0.001 vs p = 0.90 |
| Confounded group comparison | ANCOVA controlling for baseline differences | p < 0.001 vs p = 0.68 |

Every failure is a genuine analytical error with a large numeric gap, not a threshold technicality. The model produces wrong answers because it chooses wrong methods, not because it is close to right and misses by a small margin.

## Why the Failures Are Real

The verifiers check the numeric consequence of the validation step, never the process. There is no keyword matching on code, no AST inspection, no LLM-as-judge. Every check is deterministic.

In the spurious regression task, the naive and correct R-squared values differ by 57x. In the clustered treatment task, the p-values differ by four orders of magnitude. In the multiple comparisons task, the model reports 7 significant pairs when only 2 are real. These are not borderline cases.

Trajectory inspection confirms the behavioral pattern across all 30 trials. The model reads the data, runs the prompted analysis, gets reasonable-looking numbers, and reports them. It never runs a secondary diagnostic check. Not once in 30 trials.

## Technical Details

**Evaluation framework.** All tasks are built in Harbor format with sandboxed Docker environments, deterministic verifiers, and reproducible synthetic data (fixed seeds, pinned dependencies).

**Quality assurance.** Every task passes a three-way sanity check before live evaluation.
- Oracle agent (runs the reference solution) must score 1
- Nop agent (does nothing) must score 0
- Naive stub (runs the analysis without the validation step) must score 0

**Dependencies.** Python 3.11, numpy 1.26.4, pandas 2.2.2, scipy 1.13.1, scikit-learn 1.5.1, statsmodels 0.14.2, lifelines 0.29.0 (for the censored survival task).

**Model under test.** google/gemini-3-flash-preview via Harbor's gemini-cli agent.

## Project Structure

```
Post-Analysis_Eval_Suite/
├── README.md
└── experiments/
    ├── V1/                          # Round 1, 7 probing tasks
    │   ├── samples/                 # the 7 task folders
    │   ├── _build/                  # generators + eval pipeline
    │   ├── logs/                    # Gemini results (3 trials per task)
    │   └── figures/                 # pass rate visualizations
    ├── V2/                          # Round 2, 21 tasks across 4 patterns
    │   ├── P1_surface-consequence/  # 6 tasks
    │   ├── P2_cascading-multistep/  # 5 tasks
    │   ├── P3_implicit-constraints/ # 5 tasks
    │   └── P4_overconfidence/       # 5 tasks
    └── V3/                          # Round 3, final 10-task suite
        ├── tasks/                   # the 10 task folders
        ├── _build/                  # data generators
        ├── jobs/                    # Harbor run outputs
        └── REPORT.md               # full writeup
```

Each task folder follows the standard Harbor layout.

```
<task>/
├── instruction.md          # the prompt (no hints about the violation)
├── task.toml               # Harbor config
├── environment/
│   ├── Dockerfile
│   └── *.csv               # synthetic data
├── solution/
│   ├── solve.sh
│   └── solve.py            # reference oracle
└── tests/
    ├── test.sh
    └── verify.py            # deterministic verifier
```

## Running the Suite

Prerequisites include Docker Desktop and Harbor (`uv tool install harbor`).

```bash
# Sanity check a single task
harbor run -p experiments/V3/tasks/autocorrelated-residuals -a oracle -y
harbor run -p experiments/V3/tasks/autocorrelated-residuals -a nop -y

# Run Gemini eval (3 trials)
export GEMINI_API_KEY=<your key>
harbor run -p experiments/V3/tasks/autocorrelated-residuals \
  -a gemini-cli -m google/gemini-3-flash-preview -k 3 -n 1 -o jobs -y
```

## The Finding

Gemini Flash does not lack knowledge. It knows what every diagnostic technique is and can explain any of them on demand. What it lacks is the behavioral habit of applying those techniques as part of its workflow. It executes the prompted task and stops. The gap between "can explain Durbin-Watson" and "actually checks Durbin-Watson after fitting OLS on monthly data" is the gap this suite measures.

This is a systematic and reproducible result. Thirty out of thirty trials, across ten tasks testing seven different statistical assumptions, produced the same behavioral pattern. The model executes the technique. It never validates the result.

## Built With

- **Harbor** for task authoring, sandboxed execution, and trajectory capture
- **Claude Code** for building task infrastructure (generators, verifiers, oracles, Dockerfiles) from specifications
- **Python** (numpy, pandas, scipy, scikit-learn, statsmodels, lifelines) for data generation and reference solutions
- **Docker** for reproducible sandboxed environments
