# A/B Test Plan: Checkout Flow Variants

## Hypothesis
We hypothesise that variant B (single-page checkout) increases the conversion
rate over variant A (multi-step checkout).

## Design
- Pre-registered minimum detectable effect: +1 percentage point (relative ~+20%).
- Required sample size (per power analysis): 5,000 users per variant.
- Planned duration: 14 days, starting 2026-04-01.
- Decision rule: two-proportion test at alpha = 0.05, evaluated ONCE at the
  end of the planned window.
- Interim peeking is not permitted under the pre-registered plan.

## Outcome metric
Binary `converted` flag per user (1 if the user completed checkout).
