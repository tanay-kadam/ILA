# Stage 1b measurement revision

**Registered:** 2026-10-04, after the Qwen2.5-7B output diagnostic and 50-question
prompt-format screen, before the full revised baseline or any revised fine-tuning.

## Status of the original Stage 1

The original plain-text, greedy Stage 1 protocol remains a failed measurement attempt.
Its results and gates are not overwritten. No Arm A/B matrix was run under that protocol.

The repair below is a new measurement study, called **Stage 1b**. Its purpose is to decide
whether canonical Instruct-model formatting provides a stable substrate on which the
already-built natural-trace experiment can be run.

## Evidence observed before this revision

An exact-token audit of all saved Qwen2.5-7B-Instruct baseline outputs found:

- GSM8K: 85.2% reached the 6,144-token cap. Every capped output contained a complete
  `\boxed{}` answer and every capped tail met the conservative repetition-loop rule.
  The first box appeared at median token 254, followed by a median 5,890 tokens. The first
  box was correct on 91.5% of capped outputs. A first-complete-box stopping counterfactual
  gave 91.8% accuracy, 271 mean tokens, and zero unresolved examples over all 500 items.
- MATH-500: 15.8% reached the cap; every capped output met the loop rule. A first-box
  counterfactual gave 73.8% accuracy, 621 mean tokens, and 0.6% unresolved examples.
- Manual review of the stratified diagnostic packet confirmed that the capped tails were
  degenerate repetitions rather than long coherent reasoning.

The diagnostic artifacts are in `results/stage1_q7/diagnostic/` on the server.

A diagnostic canonical-chat screen on the first 50 items of each set then produced:

- GSM8K: 84% accuracy, 282.64 mean generated tokens, 0% truncation.
- MATH-500: 72% accuracy, 559.88 mean generated tokens, 0% truncation.

These 50-item results motivated this revision but will not be treated as confirmatory
evidence.

## Fixed Stage 1b protocol

### Model and data

- Base model: `Qwen/Qwen2.5-7B-Instruct`.
- Natural-trace data: the existing `data/stage1` build, including the 2,500-token trace
  limit and existing decontamination and subset assignments. No examples will be rebuilt
  or reselected for Stage 1b.

### Prompt formatting

- The semantic user prompt remains: question, the step-by-step instruction, and the
  requirement that the final answer appear in `\boxed{}`.
- The trailing plain-text `Answer:` cue is removed.
- The base tokenizer's canonical chat template is applied to one user message with
  `add_generation_prompt=True`.
- Fine-tuning and evaluation must use the same canonical chat representation. Training
  loss remains completion-only on assistant tokens, and the assistant completion ends in
  the model's end-of-turn/EOS token.

### Decoding and outcomes

- Greedy decoding remains fixed; no sampling, repetition penalty, beam search, or
  loop-aware decoding is introduced.
- `max_new_tokens` remains 6,144, with the model EOS/pad token IDs as stops.
- The primary cost remains full generated tokens through EOS, excluding EOS. Capped
  outputs remain counted at the cap and truncation is always reported.
- Tokens through the first complete `\boxed{}` answer, first-box accuracy, missing-box
  rate, and repetition indicators are secondary diagnostic outcomes. They do not replace
  the primary outcome for the Stage 1b matrix.

## Prospective gates

### Full untuned baseline

Evaluate all 500 GSM8K and all 500 MATH-500 items with canonical chat formatting.

- PASS only if truncation is below 20% on each set.
- Otherwise stop Stage 1b; do not fine-tune or launch the matrix.

### Single fine-tuned pilot

If the full baseline passes, train only `armA_random`, seed 0, using canonical chat
formatting and the existing 7B configuration.

The pilot passes only if:

- truncation is below 20% on each eval set; and
- accuracy on each set is no more than 5 absolute percentage points below its revised
  untuned canonical-chat baseline.

This non-inferiority requirement replaces the original requirement that MATH-500 accuracy
must exceed the untuned base. At 7B, preservation of a strong base is the relevant
stability check; improvement is not required to measure inference cost.

### After the pilot

Even if the pilot passes, the 29-run Arm A/B matrix does not start automatically. Report
the revised baseline and pilot results, estimate the 7B compute budget, and obtain explicit
approval before launching the matrix.

All original Stage 1 failures remain part of the project record and must be reported in
any paper or technical report.
