# Stage 1c: bounded post-SFT stabilization study

**Registered:** 2026-10-05, after the completed Stage 1b pilot failed and before any
Stage 1c training or evaluation.

## Motivation and status of prior results

The canonical-chat Stage 1b untuned Qwen2.5-7B-Instruct baseline was stable, but the
`armA_random`, seed 0 pilot trained for 460 optimizer steps (two epochs) failed: GSM8K
accuracy/truncation changed from 0.910/0.000 to 0.774/0.152 and MATH-500 changed from
0.722/0.010 to 0.368/0.510. Mean generated tokens rose about 5.3-fold on both sets.
Stage 1b remains a failed preregistered experiment and is not overwritten.

Stage 1c asks whether excessive exposure to long reflective traces, rather than the
mere presence of those traces, caused the instability.

## Hypothesis and falsification

**Hypothesis:** If excessive long-trace exposure causes the Stage 1b collapse, then an
early checkpoint from the same `armA_random`, seed 0 training recipe will retain base
accuracy and stable termination while showing nonzero supervised adaptation.

The hypothesis is falsified at the original learning rate if none of the prospectively
chosen checkpoints passes the engineering gate. One final learning-rate study at
`1e-5` is allowed; if that also has no passing checkpoint, Stage 1 natural-data work
stops. No additional hyperparameter search is authorized by this plan.

## Independent variable and controls

The first study changes only optimizer-step exposure: 15, 30, 60, 115, or 230 steps.
Model, data and ordering seed, LoRA configuration, optimizer, prompt format, decoding,
and evaluation code remain the Stage 1b values. A single continuous training run saves
LoRA adapter snapshots at those exact steps.

If no checkpoint passes, the permitted fallback repeats the same exposure study with
only the learning rate changed from `1e-4` to `1e-5`.

## Splits and leakage control

Each existing 500-example evaluation set is deterministically partitioned by a salted
SHA-256 ordering (`split_seed=20261005`) into a 100-example engineering subset and a
400-example confirmation subset. All checkpoints may be compared on the engineering
subsets. Only the largest passing checkpoint is evaluated on confirmation. The source
data, prior 13-gram decontamination, questions, and gold answers are unchanged.

These benchmarks were observed in earlier stages, so Stage 1c is a stabilization study,
not a pristine benchmark claim. Checkpoint selection is nevertheless isolated from its
400-example confirmation partition.

## Metrics and prospective gates

Primary stability metrics are last-box accuracy, full generated tokens through EOS,
truncation, and conservative repetition-loop incidence. First-box accuracy and tokens
are diagnostic secondary metrics.

A checkpoint passes a split only if, on **both** constituent datasets:

- accuracy is no more than 5 absolute percentage points below the corresponding
  canonical-chat untuned baseline;
- truncation is at most 10%; and
- conservative likely-loop incidence is at most 5%.

Among engineering-pass checkpoints, select the **largest optimizer step**. This fixed
rule favors the strongest exposure that remains stable and avoids post-hoc selection
for a desired token-cost effect. The selected checkpoint must independently pass the
same gate on the 400+400 confirmation partition.

If confirmation passes, rerun only the selected exposure using seed 1 and require the
same confirmation gate. A seed-1 failure stops Stage 1c. A two-seed pass is still only a
stability result; the 29-run Arm A/B matrix requires separate explicit approval.

## Artifacts

Save resolved configs, commands, run records, lightweight LoRA snapshots, training logs,
raw per-example generations, output diagnostics, gate reports, device identity, and
pipeline status. Failed checkpoints and runs are retained.

## Alternative explanations checked before interpretation

- Audit first-box versus last-box accuracy and post-answer tokens in the Stage 1b pilot.
- Audit training targets for boxes, post-box tails, repetition, and canonical end tokens.
- Evaluate Stage 1c snapshots as unmerged vLLM LoRAs; evaluate the seed-1 replication as
  the standard merged model, providing a bounded check that adapter merging is not the
  sole source of the observed behavior.

