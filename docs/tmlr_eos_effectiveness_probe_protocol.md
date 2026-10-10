# Frozen-checkpoint terminal-EOS effectiveness diagnostic (2026-10-09)

Posthoc diagnostic prompted by the completed control/audit, not confirmatory selection
or a new coefficient search. No training, model updates, target rewriting, new evaluation
prompt, or primary-score replacement. Original matrix and margins remain frozen.

Question: is the scalar terminal-EOS CE contribution numerically small in teacher-forced
training contexts at the base and final checkpoints? Operational screen: added objective
share R=9*sum(EOS CE)/sum(all supervised-token CE) <1% per sampled condition. R>=1%
falsifies that small-share screen for that condition. Report every condition, including
exceptions, rather than only conditions supporting saturation. This is not a gradient
norm test, equivalence test or proof of the cause of the null result.

Use both pinned base models and all18 final adapters per family from the completed EOS
matrix (weights1/10 x3 arms x3 seeds). Select32 training examples per arm by fixed
SHA256 order `probe20261009|uid|question|target`; same examples across weights, seeds and
families. No inference outcomes or gold-score changes enter sample selection. Keep
the original verified target text and canonical masked serialization/context limits.
Teacher forcing deliberately evaluates training contexts, not generalization or actual
free-running termination. Eval mode disables dropout; it does not replay training-time
dropout or measure all intermediate optimizer steps. Base probes approximate the
zero-initialized-LoRA starting function, not every historical training state.

Single-example forwards with labels, inference_mode, no optimizer, bf16 weights/FP32 CE.
Weight terminal EOS at its actual prediction position; retain Qwen whitespace suffix
labels and unchanged native supervised-token denominator. Save per-example native CE,
terminal CE, exp(-terminal CE), target-token count, and R; report sums/quantiles as
descriptive diagnostics. Numerically saturated CE/probability is not mathematical
certainty and small objective contribution need not imply small gradient influence.

Run a separate two-example base smoke per family and verify finite loss, token shape,
GPU/model path and terminal-position validation. Then process all frozen conditions.
Use leases/capacity checks and GPUs0,1,2,3,4,7 only; never5/6. Bound memory at34 GiB,
retain failures, and run detached so laptop/VPN loss does not kill the diagnostic.
New namespace: results/tmlr_eos_effectiveness_probe. Do not overwrite an existing probe.

Outputs are diagnostic only and must not be used to select a new EOS weight, change the
completed experiment's primary operational gate, imply actual training gradients were
logged, or rule out all termination mechanisms. No paper edits or acceptance claims.
