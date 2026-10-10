# Stage 1c dose-response follow-up: registered phase 1

**Registered:** 2026-10-06, after the 15-step matrix audit and before any new
dose-60 high/low-reflection run was trained.

## Scientific question

If natural-composition effects require meaningful exposure, then increasing otherwise
identical adaptation from 15 to 60 optimizer steps will produce a reproducible change
in mean generated tokens for high- or low-reflection subsets relative to the random
subset, without a material accuracy loss.

The hypothesis is falsified at this dose if neither reflection contrast produces a
credible and practically material token-cost signal under the rule below.

## Design

- Independent variable: training-subset condition (`armA_high`, `armA_low`, or
  `armA_random`).
- Dose: optimizer step 60 on the same cosine schedule planned for 230 steps.
- Seeds: 0, 1, and 2 for every condition.
- New training: high and low at all three seeds, plus random seed 2 (seven runs).
- Reuse: existing validated random seed-0 and seed-1 step-60 adapters.
- Evaluation: the same fixed GSM8K-500 and MATH-500 examples, canonical chat,
  greedy decoding, and 6,144-token limit used in the audited matrix.
- Primary metric: mean generated tokens across the 1,000 evaluation questions.
- Guardrails: accuracy and truncation rate, reported by benchmark and pooled.

Model, LoRA architecture, learning rate (`1e-5`), effective batch size, data files,
prompting, preprocessing, evaluation code, and hardware class remain fixed. Exact
evaluation-question overlap with the training source pool was previously audited as
zero. Evaluation examples and order must match across conditions.

## Registered phase-2 gate

**Execution amendment, before any new step-230 condition is trained:** The user
clarified that the remaining usage limit concerns Codex, not GPU time, and authorized
end-to-end server queuing. The signal and accuracy/truncation criteria below remain
fixed. A passing step-60 gate now automatically launches the three-condition,
three-seed step-230 matrix (seven new trainings, two reused random adapters), followed
by integrity checks and analysis. A failing gate ends with step-60 analysis. The
standalone dose-60 controller still ends at its own analysis; the separately queued
`launch_stage1c_dose_chain.sh` implements this amendment. No later stage is launched.

The dose chain launches step 230 only if at least one contrast
its dose-60 result meets either signal rule:

1. the hierarchical 95% token-difference interval excludes zero and the absolute
   relative difference is at least 2%; or
2. the absolute relative difference is at least 5% and its sign is consistent across
   all three paired seeds.

In both cases, pooled accuracy loss for the preferred lower-cost condition must not
exceed one percentage point and truncation must remain below 2%. Otherwise, stop and
report the 15/60-step low-dose null. Any step-230 continuation requires review and a
separate launch unless handled by the authorized dose-chain amendment above.

## Artifacts and claim scope

Save resolved configs, run records, adapter weights, logs, raw predictions, summaries,
and the registered analysis. Failed and partial runs are retained. This phase tests
whether the 15-step null changes by step 60; it does not establish equivalence or a
universal absence of composition effects.
