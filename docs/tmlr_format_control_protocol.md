# Experimental gap closure: AQuA format sensitivity (2026-10-08)

Scope: experiments and diagnostics only. Do not modify either manuscript, the primary
evidence packet, or the original evaluation artifacts. This is a prospective repository
protocol for an exploratory follow-up, not a registered confirmatory study. Existing
results motivated this control; AQuA is no longer an unseen evaluation set.

## Hypothesis and falsification

If explicit formatting demonstrations reduce option-value/option-letter confusion,
then the unadapted Mistral's valid boxed-letter rate will increase by at least 20
percentage points on the fixed AQuA set. An increase below that threshold falsifies
this operational hypothesis. Also report whether the rate reaches 90%; do not change
the prompt if either criterion fails. Accuracy improvements alone do not establish
improved reasoning. The separate robustness question is whether the composition-associated
cost/truncation contrasts persist with this prompt; disappearance or reversal narrows
the original interpretation and must be reported.

## Single intervention and controls

Change only the semantic prompt template, identically across both families and all
conditions. Add two invented elementary examples showing the boxed-letter convention,
plus a final-letter reminder. Retain step-by-step reasoning; do not request answer-only
generation. The intervention is the whole fixed prompt package, not an isolated estimate
of the contribution of each sentence. Input-token cost changes and is reported separately
from generated-token cost; this is a robustness check, not a new efficiency method.

Hold fixed: the original 254 AQuA UIDs/order/options/gold, all 27 adapters plus base in
each family, three training seeds per arm, pinned base models and adapter bytes,
canonical single-turn chat, greedy decoding, vLLM environment, 6144 generated-token
cap, strict boxed-choice scorer, and original high/low/random subsets. No training,
checkpoint selection, prompt search, or new hyperparameter tuning is permitted in this
matrix. Original and controlled prompts may run on different allowlisted devices of
the same hardware class; record physical GPU and versions. Original model-family
training-resource differences remain confounds for causal architecture comparisons.

## Dataset and leakage

Use the existing frozen development-held-out AQuA manifest and source bytes. Verify
UID/gold alignment and source option ordering. Existing exact/13-gram filtering is
retained; pretraining and semantic contamination are not ruled out. AQuA's five-option
format makes it appropriate for this control but insufficient for cross-domain claims.
Invented examples (1+1 and 9-4) are fixed before new predictions and contain no evaluation
answer or benchmark-derived demonstration. Base is the same-prompt reference for every
quality comparison. Reuse of this test set is explicitly exploratory.

## CPU-only option-content audit

Preserve the frozen primary scores. Map the last boxed payload to all five options
without passing gold to the mapper. Allow exact normalized content and exact rational
numeric equality only; preserve units and percent symbols. Explicit label+content must
agree with that label's option. Duplicate matches and label/content conflicts are
ambiguous, not correct. Do not extract a convenient number from reasoning, solve
expressions, apply fuzzy similarity, or recover absent boxes. Map first, compare gold
second. Report strict correctness, uniquely content-mapped correctness, wrong mapped
choices, ambiguous, and unparseable outputs separately, plus truncation-stratified
counts. Automated content mapping is a diagnostic, not validated human scoring.
Save deterministic category-stratified cases for subsequent human review; do not label
the exported sample as reviewed.

## Metrics, seeds, uncertainty, and failure policy

Evaluate 56 checkpoints (28 per family), AQuA only. Report every seed, valid-letter rate,
strict accuracy, mean capped generated tokens, cap-hit rate, conservative loop rate,
tokens per correct answer, finished length, and exact descriptive cap-frequency/finished-
length decomposition. Three existing training seeds permit comparison with the original
matrix but give limited power for small effects; base has no training-seed uncertainty.
Retain 2000-draw independent-arm seed/shared-question bootstrap and labeled paired-seed
sensitivity from the original analysis. Also report original-versus-control within-checkpoint
differences, without pretending checkpoints are independent replicates. Do not treat the
full 27-checkpoint pooled average as a hypothesis-test sample size.

For quality, use each model's same-prompt base, accuracy loss <=5 percentage points,
truncation <2%, and loop rate <=2% per seed. These are diagnostics, not a rule for excluding
unfavorable runs. Report every condition, including failed quality gates and null or
reversed contrasts. The prompt-validity hypothesis is the primary new test; remaining
contrasts are exploratory with no multiplicity-adjusted confirmatory claim.

## Execution and artifacts

Use physical GPUs 0,1,2,3,4,7 only; never 5 or 6. Do not preempt other users. Use shared
GPU leases and capacity checks. Before full evaluation, run separate two-question base
and adapter smokes for each family; verify device logs, resolved config, fingerprints,
token counts, and strict scores. Smoke outputs have separate directories and cannot be
mistaken for full metrics. Launch detached, retaining PID, commands, status, failures,
logs, config/data/code hashes, raw predictions, summaries, and automatic final analysis.
New roots are results/tmlr_format_control_{qwen,mistral} and
results/tmlr_format_control_controller. Refuse existing output roots on initial launch.
Do not silently rerun failed jobs or overwrite old results. Any incomplete matrix is
reported as incomplete, not ready for analysis.

## Remaining priorities after this bounded control

1. Termination-mechanism intervention: a fixed, matched terminal-EOS loss-weight control
   with native-loss/gradient equivalence tests before any training. It could distinguish
   termination supervision from general representation changes, but would not establish
   reflection as a causal mechanism. Requires a separate prospective design.
2. Actual decoding-cap intervention on fixed representative base/stable/degraded
   conditions: shorter-cap clipping of existing sequences is descriptive, not evidence
   about what would finish at a larger cap. Avoid presenting clipping as new generation.
3. Matched training-budget/confound controls: natural composition arms differ in trace
   content, length, and possibly problem mix. A causal claim about reflection needs
   question-matched or length/optimization-dose-matched interventions; otherwise retain
   the narrower composition-associated boundary claim.
4. Human validation of the repetition heuristic and ambiguous answer mappings using a
   deterministic, blinded sample. Exporting cases is not a substitute for human review.

More runs are not automatically stronger evidence. Prioritize controls that can change
the interpretation; report unfavorable outcomes without outcome-driven redesign.
