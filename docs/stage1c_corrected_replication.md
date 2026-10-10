# Stage 1c corrected seed-1 replication addendum

**Registered:** 2026-10-06, after auditing the completed Stage 1c pipeline and before
the corrected replication was launched.

## Why the recorded replication is not decisive

The Stage 1c seed-0 candidate was optimizer step 15 from a run whose cosine schedule
was planned for 230 steps. The pipeline's seed-1 replication instead set
`train.max_steps=15`, which compressed the cosine schedule: at step 10 seed 0 used
approximately `9.998e-6`, while seed 1 used approximately `3.887e-6`. The evaluations
also differed: seed 0 used the saved adapter through vLLM's dynamic-LoRA path, while
seed 1 used a merged model. Thus the observed seed gap did not isolate training seed.

The original pipeline status and artifacts remain unchanged and must not be deleted or
rewritten.

## Corrected replication

- Model/data/format: unchanged from Stage 1c.
- Learning rate: `1e-5`.
- Training seed and data-order seed: 1.
- Planned training duration and cosine schedule: 230 optimizer steps, exactly matching
  the seed-0 trajectory run.
- Saved adapter checkpoints: steps 15, 30, 60, 115, and 230.
- Evaluated checkpoint: step 15 only, chosen before this run based on the seed-0 gate.
- Evaluation representation: unmerged step-15 LoRA through the same vLLM dynamic-LoRA
  path used for seed 0.
- Evaluation data: the previously locked 400-example GSM8K and 400-example MATH
  confirmation partitions, with identical prompt and greedy decoding.

No result from steps 30/60/115/230 will be used for recipe selection. They are saved
only because completing the identical 230-step schedule is the simplest way to ensure
the step-15 scheduler state is identical without changing training machinery.

## Fixed gate

On both confirmation datasets:

- accuracy must be no more than 5 absolute percentage points below the canonical-chat
  base;
- truncation must be at most 10%; and
- conservative likely-loop incidence must be at most 5%.

If seed 1 passes, the `1e-5`, step-15 recipe is considered replicated for the stability
gate only. If it fails, Stage 1c stops. In either case the 29-run matrix does not start
without separate explicit approval.

