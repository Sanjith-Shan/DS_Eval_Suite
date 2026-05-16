# Design Pattern for Headroom Against Frontier Flash Models

**Author:** Sanjith Shanmugavel
**Date:** 2026-05-15
**Source experiment:** seven-task data-science evaluation suite, run against
`google/gemini-3-flash-preview` via Harbor 0.7.0, three trials per task.
Aggregate result: 6 of 7 tasks at pass@3 = 1; 1 task at pass@3 = 0
(deterministic). Full data in `report/report.md`, transcripts in `logs/`.

---

## 1. The problem this document solves

The take-home brief required tasks where a frontier flash model would pass
fewer than 30% of three attempts. Six of my seven tasks fell well short of
that bar — `gemini-3-flash-preview` solved them cleanly. The seventh, by
contrast, produced a textbook failure across all three attempts, with
identical numeric output every time.

The interesting question is not "why were six too easy" in isolation; it's
**what is structurally different about the one task that worked, and how do
I generate more like it?** This document captures the pattern, the evidence
behind it, and a concrete recipe for designing additional tasks that exploit
the same gap.

The intended reader is me-in-three-months, or anyone else picking up the
eval project, who wants to grow the suite without re-running the trial-and-
error of v1.

---

## 2. The finding, stated plainly

> **Frontier flash models reliably solve problems that have a name. They
> reliably miss the second-order step that follows the named fix, when that
> step is not itself named.**

A "named" problem is one with a Wikipedia page, a Stack Overflow tag, a
textbook section, a blog-post title, or a benchmark category. Examples from
my suite that the model passed: Simpson's paradox, sequential peeking in
A/B tests, ANOVA assumption violations, daylight-saving non-existent times,
structural breaks in time series, spurious correlation under a confounder.
Every one of these has a name and a canonical fix written down somewhere on
the internet.

An "unnamed" step is one that follows from the named fix but is not itself a
named technique. It's the kind of move a senior practitioner makes by
habit — *re-examining their own work after the named fix is applied* — but
which nobody bothers to label, because it's "obvious" once you've done it
a few times.

The pattern I want to exploit is: **the model has read about every named
problem, but it has not internalised the habit of looking again.**

---

## 3. The evidence — six pass, one fails

Six tasks in the v1 suite passed every trial against the target model. They
covered:

- Causal vs. correlational interpretation (two tasks)
- Experiment-protocol violations
- Statistical-test assumption checks
- Time-series structural breaks
- Production ETL with timezone & schema drift

All six follow the same shape: the instruction describes a scenario, the
canonical failure mode for that scenario is named (or unambiguously implied)
in the data, and the agent has to apply a known textbook fix. The model
recognises the pattern, applies the textbook move, and the verifier passes.

The one task that produced reliable failure was `data-leakage-detection`.
It asks the agent to audit a buggy machine-learning training script and
produce a corrected version. The script has three planted leakage bugs.
The fix for each individual bug is well-known and well-named (split before
preprocessing; encode targets using training data only; do feature selection
on training data only). **The model fixed all three correctly, in all three
attempts.**

What it did not do — also in all three attempts — was the unnamed step that
follows the third fix. The third fix, applied to my deliberately-shaped
data, converts one input column from a near-perfect cheat sheet into
structured noise. The senior-DS move is to look at the column post-fix,
recognise it has lost all informative content, and drop it. The model
applied the named fix and shipped. Final test accuracy: 0.681 in every
trial, just below my 0.70-0.85 verifier band. The reference solution drops
the column and reaches 0.745.

The failure is not noise: three trials, three identical-to-the-decimal
results. It is a systematic blind spot, not a sampling artefact.

---

## 4. Why this pattern exists

A frontier flash model's "knowledge" is, mechanically, a compression of an
enormous amount of text. The training corpus is dense with material like:

- Stack Overflow answers ("Q: I'm getting suspiciously high accuracy. A:
  You probably have data leakage. Here's how to check.")
- Textbook chapters ("Simpson's paradox is when the aggregate result
  contradicts the stratified one. Here's the canonical example.")
- Blog posts ("Five common pitfalls in A/B testing. #1: Peeking…")
- Library documentation ("`tz_localize(nonexistent='raise')` raises an
  error on missing local times. Use `'shift_forward'` to map them to the
  next valid instant.")
- Benchmark write-ups ("Frontier models routinely fail at X because…")

All of that material is **organised around named concepts**. The name is
the index. The fix is the entry. Train a model on a billion of these and it
becomes excellent at "I see X → I do Y," where X is a name.

What the corpus contains far less of — vanishingly less — is text that
articulates the *habit* of looking again. Things like:

- "After you fixed the leak, did you check whether the column still
  carries any signal?"
- "After you converted timestamps to UTC, did you check that nothing went
  to NaT silently?"
- "After you applied the Bonferroni correction, did the pair you cared
  about actually survive?"

These aren't blog-post-worthy. They're code-review comments. They're the
tone of voice a senior engineer uses on a pull request. They are not in the
corpus the way named patterns are, because they aren't *content*, they're
*disposition*.

The model therefore graduates from the named-fix step very capably and
stops there. That is the seam.

---

## 5. The design recipe

The recipe is three steps. Every task in v2 of this suite, and any future
suite built for the same model tier, should follow it.

### Step 1: Front-load with a named problem

Pick a setup that has a well-known failure mode. The model should be able
to *name the failure mode* from a paragraph of context. Examples:

- "This pipeline reports suspiciously high accuracy" → data leakage.
- "These groups have unequal variances" → ANOVA assumption violation.
- "The aggregate result disagrees with the subgroup results" → Simpson's
  paradox.
- "This experiment was stopped early" → peeking / sequential testing.

The model should solve the named problem. Do not try to make the named
problem itself hard — the model will get this right, and the verifier
should let it.

### Step 2: Engineer an unnamed side effect of the correct fix

This is the actual design work. The setup must be constructed so that the
correct fix to the named problem produces a **second consequence** that is
not itself a named pattern in the training corpus. The second consequence
should be:

- A natural result of the correct fix (not a separate bug planted on top).
- Visible only if the agent examines its work after applying the fix.
- Resolvable by a routine action a senior practitioner would take by
  habit (drop a column, flag a row, widen an interval, re-check a join).
- Not described anywhere as a standard step. No blog post says "after
  fixing X, always do Y."

In my data-leakage task, the second consequence is that the
correctly-fixed `customer_segment` column becomes structured noise.
Dropping it gets you back into the verifier's pass range; keeping it
doesn't. There is no blog post titled "After fixing target-encoding
leakage, audit your encoded features for collapsed variance" — and even
if there were, the specific data condition that triggers it (very small
group sizes) isn't a standard discussion point.

### Step 3: Make the verifier check the side effect, not the named fix

The verifier should not check whether the named fix was applied. The
verifier should check the *consequence* — the thing the agent only gets
right if it did the unnamed audit step.

For the data-leakage task, the verifier doesn't grep for "fit_transform"
or "train_test_split" patterns. It runs the fixed pipeline and checks the
resulting accuracy lands in [0.70, 0.85]. A pipeline that fixed the
leakage but kept the dead feature lands at 0.681 and fails. A pipeline
that fixed the leakage and dropped the dead feature lands at 0.745 and
passes. The named work is invisible to the verifier; the audit is what's
measured.

This is the part that's easiest to get wrong. If the verifier checks the
named fix directly, the task degenerates to "do the named thing" and
becomes a pass-3/3 task again. If the verifier checks something too far
removed from the named fix, the task becomes solvable by accident or by
an unrelated route. The verifier band must be set so that *only* the
named-fix-plus-audit path lands in it.

---

## 6. Worked examples — applying the recipe to other DS subareas

These are concrete tasks I would build for v2. None of them are in the v1
suite. Each follows the three-step recipe and should produce headroom
against the target model.

### Example A: Time-series outlier handling with hidden survivor

**Step 1 (named).** Give the agent a daily revenue time series with a few
obvious outliers (one day at 10× the median, another at 0). Ask them to
clean the data and produce a forecast.

**Step 2 (unnamed side effect).** Engineer the data so that after removing
the obvious outliers, one *day-of-week* effect becomes negative — Sunday
revenue is consistently lower than every other day, but Sunday only has 13
observations in the cleaned data, none of which were the outliers. The
correct fix is not just "remove outliers"; it is also "recognise that
Sunday is now underpowered to fit independently and either pool with
Saturday or model it as a random effect." Nobody calls this a named pattern.

**Step 3 (verifier).** The verifier doesn't check that outliers were
removed. It checks forecast accuracy on a held-out Sunday in the test
window. A model that removes the outliers and fits weekday × outcome
independently produces a Sunday forecast with implausibly wide confidence
intervals; the verifier rejects forecasts whose Sunday CI exceeds a
threshold.

### Example B: Multiple imputation that destroys a relationship

**Step 1 (named).** Give the agent a dataset with 12% missing values in a
single column and ask them to impute and run a regression. Missingness is
clearly stated; standard practice is mean imputation or multiple imputation.

**Step 2 (unnamed side effect).** Engineer the missingness so it is
*correlated with the outcome* (MNAR). Mean imputation collapses the
relationship between the imputed column and the outcome, making a real
predictor appear non-significant. The correct fix is to recognise MNAR
from a quick missingness-vs-outcome plot, then use a missingness indicator
or an explicit MNAR model. Nobody calls this a named pattern in the way
"missing data → impute" is named.

**Step 3 (verifier).** The verifier checks that the regression coefficient
on the formerly-missing column is statistically significant and within
±0.1 of the true coefficient. Mean imputation produces a non-significant
coefficient and fails. Recognising MNAR and using an indicator passes.

### Example C: Feature-engineering audit after one-hot encoding

**Step 1 (named).** Buggy script one-hot-encodes a categorical column with
high cardinality before splitting. Standard fix: encode after split, handle
unseen categories in test.

**Step 2 (unnamed side effect).** Engineer the categorical so that after a
correct fix, ~30% of the one-hot columns are nearly all-zero in training
(rare categories that happened to land in test). These columns are
effectively dead. The correct response is to filter rare categories before
encoding (a common-but-unnamed practice). Without this filter, the model
trains on a sparse, mostly-empty matrix and over-fits the few non-zero
entries.

**Step 3 (verifier).** The verifier runs the fixed pipeline against a
held-out evaluation set drawn from the *training distribution*. A pipeline
that encoded all categories produces poor evaluation accuracy; one that
filtered rare categories first passes. The verifier doesn't inspect the
filter — it inspects the consequence.

### Example D: Stratification across two dimensions

**Step 1 (named).** Hospital outcome data, two treatments. Aggregate
suggests B is better. Severity-stratified analysis suggests A is better in
both strata. This is the v1 Simpson's-paradox task, and the model passes.

**Step 2 (unnamed side effect).** Engineer the data so that stratifying on
severity *alone* gives the right answer (A) but stratifying on severity ×
age-band gives a different answer (B is genuinely better for elderly
patients within each severity tier). The correct response is to recognise
the second-axis interaction; the single-axis stratification is necessary
but not sufficient. There is no standard name for "always check whether a
second stratification reverses your conclusion."

**Step 3 (verifier).** The verifier checks not just `better_treatment`
but also a `subgroup_recommendations` field that names age × severity
combinations where treatment B should be preferred. Single-axis
stratification produces an empty subgroup field and fails.

---

## 7. Verifier patterns that work, and ones that don't

The verifier is where most of the design effort goes. Some patterns I
trust, some I don't.

**Patterns that work**

- **Numeric bands.** "Output accuracy must be in [0.70, 0.85]." Forces the
  consequence to be reached, doesn't care about the path.
- **Required-field presence.** "Output JSON must include a non-empty
  `subgroups_with_opposite_conclusion` array." Forces the audit step.
- **Re-running the agent's code.** "Import the agent's fixed script, call
  its main function, check the return value." Robust against the agent
  doing the work in a different way than I expected.
- **Hidden ground truth in `/tests/`.** "Compare forecast to held-out
  actuals." The agent can't game what it can't see.

**Patterns I don't trust**

- **Source-code regex.** "Verifier passes if the script contains the
  substring 'shift_forward'." Easy to game, brittle to rephrasing.
- **Keyword search in explanations.** "Verifier passes if the agent's
  prose mentions 'Simpson's paradox'." The model will mention the
  keyword without understanding it; or will solve correctly without using
  the exact word.
- **LLM-as-judge with no anchor.** "Have another model evaluate whether
  the analysis is correct." Noise floor is too high to distinguish
  good-but-incomplete from good-and-complete at n=3.

The verifier should test the **substance** of the conclusion — usually a
numeric or structural property of the output — and ignore the journey the
agent took to get there.

---

## 8. Failure modes when designing this way

Designing audit-step tasks is harder than designing named-failure tasks.
The mistakes I'm most likely to make:

**Side effect too subtle.** If the engineered side effect produces a
1-point accuracy drop, the verifier band has to be so tight that random
variation between trials starts producing flaky pass/fail behaviour. The
side effect needs to produce a robust signal — for the data-leakage task,
the gap between "fixed but kept dead column" (0.681) and "fixed and
dropped dead column" (0.745) is large enough that a 0.70 floor is
reliable.

**Side effect too obvious.** If the engineered side effect is itself a
named pattern in the corpus, the model recognises it and you're back to
two-named-problems instead of one-named-plus-one-unnamed. The audit step
needs to be a routine senior-practitioner action, not another textbook
move.

**Verifier checks the named fix.** Easy to slip into — natural to write
"verifier passes if the fixed pipeline doesn't have `fit_transform`
before `train_test_split`." Don't do this. Verify the consequence, not
the procedure.

**Oracle solution doesn't exercise the audit step.** If your reference
solution can pass the verifier *without* the audit step, the task
doesn't actually measure the audit step. Run the oracle, confirm it
exercises the move you care about. Then write a "passes the named fix
but skips the audit" stub and confirm *that* one fails the verifier. If
both pass, the verifier isn't tight enough.

**The "named" front-loading is actually harder than expected.** Watch
for tasks where the named problem itself trips the model — that adds
noise. The first half should be a near-100% pass for the model so that
the verifier outcome is dominated by the audit-step decision.

---

## 9. How to scale this from one task to a hundred

Each of the four worked examples above is a *template*, not a task. A
template has knobs: which feature is the dead one, how many groups are too
small, what the missingness pattern looks like, which second-axis variable
flips the conclusion. Each combination of knobs is a fresh task with the
same structural shape.

For the v2 suite I'd target ~20 templates × ~10 parameter combinations =
200 tasks. The QA loop on each template:

1. Build the template, write one parameter combination by hand.
2. Run `harbor run -a oracle` → reward 1.
3. Run `harbor run -a nop` → reward 0.
4. Run `harbor run -a gemini-cli -m google/gemini-3-flash-preview -k 3`.
5. If pass@3 = 1, the template is too easy — go to step 6.
6. If pass@3 = 0 reproducibly (same wrong answer in multiple trials),
   the template is good. Sample 9 more parameter combinations.
7. If pass@3 is somewhere in between, that's the sweet spot — random
   variation between trials. Sample 9 more combinations.

This is what step 3 of the QA loop in `report/report.md` §4 didn't catch
in v1: I should have piloted *each* template against Gemini before
shipping all seven, instead of trusting prior probabilities from the
literature.

---

## 10. The one-sentence version

> *Design every task so that the front half is a problem the model has a
> name for, and the back half is an audit the model has no name for, and
> make the verifier check only the back half.*

That sentence captures everything in this document. If you can produce
tasks that satisfy it, you produce headroom against any model that learns
primarily from named-pattern content — which is, for the foreseeable
future, every frontier flash model on the market.
