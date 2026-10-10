# Terminal-EOS supervision control (2026-10-08)

Prospective repository protocol for an exploratory follow-up, not public preregistration.
No manuscript edits. Prior results motivated the intervention; evaluation sets are reused.

Hypothesis: increasing only terminal-EOS cross-entropy weight from 1 to 10 will reduce
cap-hit frequency in the fixed 1e-5/60-step degraded adaptation conditions without a
greater-than-five-percentage-point accuracy loss relative to their weight-1 counterpart.
Operational primary support requires a >=10 percentage-point reduction in pooled
cap-hit frequency per family (equal weights across three arms/seeds, and separately
equal weights across SVAMP/AQuA), a hierarchical CI excluding zero, and accuracy loss
<=5pp separately on both benchmarks. Failure of any criterion does not support this
operational hypothesis for that family. Report every arm/seed even if pooled criteria
fail; no coefficient tuning or new recipe search follows a negative result.

Intervention: native causal token loss plus 9 times the CE for predicting the last
supervised EOS token of each example, divided by the same native supervised-token
denominator. Do not upweight internal EOS tokens, mask prompt differently, normalize by
sum of weights, rewrite traces, alter targets, or use answer-only decoding. Weight 1
must exactly follow native Trainer loss and gradients, including accumulated-batch token
normalization. The additional term modestly changes loss/gradient scale; clipping and
stronger stopping priors remain alternative explanations, not bugs to hide.

Amendment before full runs (2026-10-09): older Qwen snapshot producers lack modern
data/code fingerprints. Add fresh weight1 comparators to avoid historical pipeline
confounding. Missing provenance is not evidence of a data bug. No new-control outcomes
motivated this amendment.

Matrix: Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3; high/low/random; seeds 0/1/2;
weights 1 and 10; fixed original LR1e-5, stop60 on planned230 cosine schedule. Thirty-six
new trainings. Fresh weight1/weight10 pairs are primary; historical counterparts are
retained as context, not substituted into the paired test.
Trace evaluation-only aliases to the actual adapter-producing training record. Copy
that resolved config; freeze current data/model fingerprints and library versions in
fresh pairs; record historical drift or missing fingerprints. Change only
experiment/output names, terminal-EOS weight, and stop-at60 where the original source
continued beyond its evaluated step60 snapshot. The first60 optimizer steps remain
matched; retain the original planned horizon and snapshot schedule. Qwen and Mistral
resource differences remain; within-family controls
are matched, not a causal architecture experiment. Base references use original prompts.
Set save_merged_model=false in both fresh weights: this changes storage only, not
optimization. V2 was stopped during smokes before full runs because historical Qwen
configs inherited merged saving and could leave adapter_dir unset. Its weight1 eval
smoke is invalid for adapter verification and is excluded. V3 requires a nonempty
existing trained adapter and prevents base-model fallback. Preserve both earlier attempts.

All original questions, target text, tokenization, LoRA configuration, batch/accumulation,
optimizer, seed, and train/eval split remain fixed. Validate all examples: zero truncation,
masked prompt prefix, final supervised EOS with only serialized whitespace following it.
V1 precheck failed before training because it assumed EOS was the final label; Qwen's
canonical template includes a newline after EOS. V2 preserves that suffix and weights
EOS at its actual causal prediction position; suffix labels keep their original weight.
Validate the decoded suffix as whitespace only, recording its permitted token IDs.
Stop rather than
silently modifying data if validation fails. Reuse archived 400 SVAMP and 254 AQuA rows,
manifest hashes, exact/13-gram exclusions and original canonical prompts. These are math
domains and reused developmental benchmarks, not pristine tests or contamination-free
foundation pretraining. No checkpoint selection from test outcomes is permitted.

Evaluate only final60 adapters on both sets with original greedy vLLM, cap6144, strict
numeric/boxed-letter primary scores. Preserve the separate fixed-format prompt study;
do not substitute that prompt or content-rescored accuracy in this mechanism experiment.
Save token IDs, native run records, wrapper/loss source hashes, config/data fingerprints,
optimizer steps, device, commands, versions and failures. Baseline formatting limitations
on Mistral AQuA remain; accuracy is explicit-letter performance, not isolated competence.

Report original/control per-seed accuracy, cap-hit, conservative loop rate, capped output
tokens, finished length and cost decomposition. Pair seed/arm/question when comparing
weight10 to weight1 (same training seed and data); resample seeds within each arm and
shared questions within each benchmark, 2000 draws, RNG20261008. For cross-arm cost
contrasts retain independent-arm seed/shared-question bootstrap and labeled paired
sensitivity. Three seeds give limited small-effect precision, not proof of equivalence.
Avoid declaring universal success if model/benchmark/arm results disagree. Quality gates
against same-prompt base remain accuracy drop <=5pp, truncation<2%, loops<=2%; reductions
in caps alone do not establish usable adaptation or an efficiency method.

Execution: allow physical GPUs 0,1,2,3,4,7 only, shared leases, capacity checks, no other
user preemption. Require CPU unit tests, Trainer accumulation equivalence checks, one
optimizer-step GPU train and two-question adapter eval smoke per family before the full
queue. Detached controller runs train->eval->integrity->analysis automatically. Failures
are retained and do not silently become completed runs. New results namespace is
results/tmlr_eos_control_v3; earlier precheck/smoke attempts are preserved separately. Never
overwrite original runs or restart an existing controller
without inspecting artifacts and documenting a separate versioned retry.

Interpretation: a positive intervention probes sensitivity to termination supervision;
it does not establish that insufficient EOS learning was the sole cause, that reflection
causally drives cost, or that the original pipeline was buggy. A negative intervention
is reported and does not trigger weight search. Question/length/dose matching and blinded
human validation remain separate experimental gaps, not completed by this control.
