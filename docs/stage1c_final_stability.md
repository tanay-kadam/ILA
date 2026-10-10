# Final bounded stability attempt (2026-10-07)

The completed 15- and 60-step vLLM results remain valid. Audit fixes do not require
regenerating those predictions. Corrected interpretation is written separately as
`analysis_v2`, preserving the original reports and registered decisions.

## Hypothesis and control

If excessive adaptation strength drives the observed looping, reducing learning rate
from 1e-5 to 3e-6 at step 60 will preserve baseline-relative accuracy and termination.
Only learning rate changes. The prior 1e-5/60-step runs are the matched recipe control;
model, datasets, seeds, LoRA, effective batch, planned 230-step scheduler, canonical
prompt, loss, decoding, token cap, and scoring are held fixed.

This is one candidate recipe, with no automatic additional learning-rate search,
step-230 phase, or Arm B matrix. It is falsified as a stable candidate if any seed-0
condition fails the engineering gate below. A stable recipe is necessary but does
not by itself establish useful inference-cost attribution.

## Phase 1: engineering gate

Train random, high, and low reflection at seed 0 and evaluate only the fixed
GSM8K-dev100/MATH-dev100 partitions from the earlier deterministic split. For
each condition and each benchmark require accuracy no more than five absolute
percentage points below its canonical base, truncation strictly below 2%, and
conservative likely-loop incidence at most 2%.

If any condition fails, save the actual metrics and a `STOP_QUALITY_FAILURE` report.
End this attempt and recommend shelving the original usable-cost claim.

## Phase 2: fixed three-seed comparison, conditional on all engineering passes

Train seeds 1 and 2 for all three conditions. Reuse only phase-1 seed-0 adapters
from this same recipe. Evaluate every run on the fixed GSM8K-500 and MATH-500
sets and separately slice the 400-example confirmation partition. Require the same
per-benchmark quality gates for every seed on confirmation, report seed variability,
and use the existing hierarchical bootstrap contrasts against random for token costs.

Primary effect: mean generated tokens per problem, pooled equally over benchmarks.
Signal requires a 95% hierarchical interval excluding zero with at least 2% absolute
relative difference, or at least 5% difference with all three seed signs agreeing.
Relative accuracy loss of the cheaper condition must be at most one percentage point.
The new analysis additionally enforces baseline-relative accuracy and truncation on
each run and benchmark. A quality failure and a stable no-signal result have separate
labels. Report the lower bound nature of token costs when censoring remains.

## Leakage and interpretation

Use existing decontaminated training data and fixed UID-aligned evaluation splits;
no preprocessing is refit on test data. All 500-example benchmarks have already been
observed, including their 400-example partitions. Confirmation here is a procedural
engineering check, not pristine held-out evidence; publication claims need independent
evaluation. Do not compare new results against old step 15 as a pure dose effect,
because learning rate also differs.

## Bound and artifacts

At most nine new 60-step trainings plus one one-step smoke training; no retraining
of the old 29-run matrix. All jobs run detached with raw outputs, generated token IDs,
configs, data/code/adapter fingerprints, library versions, and status/log files.
Smoke testing and explicit CUDA verification precede full training. No overwrite of
old artifacts is allowed. GPU 3 is selected because other authorized GPUs are in use.
The pipeline ends with a gate report or full comparison analysis, even for null results.

Potential alternatives to a positive result include different problem composition,
shared benchmark adaptation, seed noise, backend numerical differences, and capped
generation. Source/length matching, three seeds, raw-output checks, base-relative gates,
and explicit caveats reduce these risks; they do not prove the exact causal mechanism.
