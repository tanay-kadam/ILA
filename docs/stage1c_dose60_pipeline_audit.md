# Dose-60 pipeline audit

Audit date: 2026-10-07. Scope: nine completed runs, 18 evaluations, 9,000 raw
predictions, source run records, training logs, checkpoint manifests, and gate code.
Original results and registered thresholds were preserved. Separate diagnostic
artifacts were created; no training or continuation jobs were launched.

## Verdict

The stop decision is correct under the registered rule. There is no detected
artifact, aggregation, scoring, scheduler, or checkpoint-selection error that
explains its truncation failure. The result is a **composition signal in a degraded
generation regime**, not a dose-60 null effect.

Both contrasts pass their signal criteria and their relative-accuracy guard. Both
fail their truncation guard. Their cheaper condition actually has higher accuracy:
random beats high reflection by 10.80 percentage points; low reflection beats
random by 11.83 points. The earlier explanation that accuracy caused the gate failure
was imprecise: accuracy is badly degraded relative to base, but that comparison is
absent from the implemented gate.

## Independent quantitative checks

Across three seeds and both benchmarks:

- High reflection: mean 4,210.16 tokens, accuracy 35.17%, truncation 62.90%,
  conservative likely-loop rate 62.20%.
- Random: mean 3,553.21 tokens, accuracy 45.97%, truncation 49.77%, loop rate 48.93%.
- Low reflection: mean 2,802.18 tokens, accuracy 57.80%, truncation 35.30%,
  loop rate 34.40%.
- Canonical unadapted base: pooled accuracy 81.60%, truncation 0.50%.

High versus random costs +18.49%, with hierarchical 95% CI [+388.10, +901.80]
tokens. Low versus random costs -21.14%, CI [-1,012.47, -516.09]. Signs agree
across all three seeds. These are differences in capped generation cost, accompanied
by major changes in failure propensity; they do not establish an efficient usable model.

For high versus random, the gate checks maximum pooled truncation 62.90% against
2%. For low versus random, it checks 49.77% against 2%. Independently recomputed
values exactly reproduce the saved gate fields.

## Verified implementation details

- All per-benchmark metrics independently match raw prediction rows. Predictions
  and correctness were recomputed from output text and gold answers.
- All evaluation UIDs and gold labels match the baseline, in the same order.
- All capped rows have the recorded 6,144-token count; no rows exceed the cap.
- Evaluation summary adapter and model paths match the corresponding run records.
- Seven new trainings stop at step 60; two reused adapters are actual step-60
  snapshots listed in their source manifests. The fact that their parent runs
  completed 230 steps does not mean the evaluated adapter came from step 230.
- All nine source runs use canonical chat, LR 1e-5, planned cosine horizon 230,
  rank 64/alpha 128, the same seven LoRA projection families, batch size 2,
  accumulation 8, and length limit 3,072. No truncated training examples are recorded.
- Source training libraries agree across seeds. All recorded losses, gradient norms,
  and learning rates through step 60 are finite. Logged LR at step 10 is
  9.998015e-6 and at step 60 is 8.717295e-6 in every run: there is no observed
  shortened-schedule mismatch in the reused adapters.
- Local and server training, evaluation, and scoring files have identical SHA-256 hashes.
- Transformers 5.18.0 (training) and 4.57.6 (evaluation) produce identical tested
  canonical prompt strings and token IDs. EOS is 151645; padding is 151643.
- In 16 sampled targets per arm, all prompt labels are masked, all assistant EOS
  tokens are supervised, and no examples are truncated. The previous complete
  random-target audit also recorded EOS near the end of every example.
- The relevant prompt, chat-dataset, output-diagnostic, Stage 1 analysis, and
  checkpoint-selection tests pass: 21 tests in 7.28 seconds. This does not replace
  missing dedicated tests for the new dose-60 guardrail code.

## Failure mechanism evidence

Nearly every truncated output is also flagged as strongly repetitive. No complete
box appears in 61.57% of high, 49.13% of random, and 35.10% of low outputs.
First-box accuracy remains essentially as poor as recorded accuracy: 34.87%,
45.37%, and 57.40%, respectively. Recovering the last complete box changes high
accuracy by only 0.067 percentage points and changes the other arms by zero.
Answer extraction cannot explain the large accuracy and truncation losses.

A separate HF/PEFT diagnostic explicitly loaded the same high-reflection seed-0
adapter and generated the same GSM8K question to the original 6,144-token cap.
It also produced 6,144 tokens without EOS and a strong repetition loop (tail unique
4-gram ratio 0.098; compression ratio 0.057). The saved vLLM output for that
question likewise reached the cap. Their wording diverged after six matching tokens,
so this is not bitwise backend equivalence; it demonstrates that vLLM alone is not
necessary for this adapter's failure. One example cannot establish full benchmark parity.

The inference is that longer adaptation pushes this recipe into repetitive reasoning
and weak termination, with severity related to the subset condition. This behavioral
finding does not identify the exact training mechanism. Long reflective targets,
adaptation strength, and preservation of the instruction model's termination behavior
remain plausible contributing factors.

The previous engineering gate had already rejected the random step-60 checkpoint:
GSM8K-dev truncation 40%, MATH-dev truncation 62%, with accuracy losses of 35 and
41 percentage points. Step 15 passed. The later matrix adds condition comparisons,
but the random-condition collapse was foreseeable and should have been highlighted
before the follow-up was proposed.

## Confirmed issues, separated from causes of this stop

1. **Misleading outcome label.** `launch_stage1c_dose_chain.sh:27` emits
   `COMPLETED_DOSE60_NULL_ANALYSIS_READY` for every gate failure. A quality failure
   after a significant signal is not a null. The design note likewise says to report
   a null whenever continuation fails. This affects scientific interpretation, not
   whether the controller correctly stopped.
2. **Incomplete quality protection.** `analyze_stage1c_dose60.py:89` compares the
   cheaper adapted model against the other adapted model. Both can collapse while
   passing this accuracy check. Baseline-relative, per-benchmark accuracy protection
   is missing. It did not change this decision because truncation independently failed.
3. **Latent HF adapter bug.** `evaluate.py:195` passes only `model_dir` into the HF
   generator. With an adapter-only run, this evaluates the base model without loading
   the recorded adapter. These 18 evaluations used vLLM, so this defect did not cause
   their observed failure. The separate audit diagnostic explicitly loads PEFT.
4. **Weak cache/provenance validation.** Existing summaries are skipped without
   comparing config, data, or adapter hashes. Evaluation records omit generated token
   IDs and backend package versions. The artifact checks found no observed stale-data
   mismatch here, but original-time code/data immutability cannot be proven from all
   run records alone.
5. **Coverage gap.** The eight tests cited during launch exercise the older Stage 1
   analysis, not the new dose-60 guardrails. Independent recomputation now checks
   the actual decisions, but dedicated regression tests for signal-versus-quality
   outcomes should precede another launch.

## Recommended action

Keep the stop decision and preserve the results as a degraded-regime composition
finding. Correct reporting labels, add baseline-relative accuracy protection, and fix
the HF adapter route before reuse. Select any next training recipe using engineering
data and a stability gate before spending on another full matrix. Do not relax the
2% truncation threshold after observing these results to manufacture a pass.

## Claim classification

- **SUPPORTED:** The registered continuation gate fails on truncation; independent
  raw-output recomputation exactly reproduces the failed guardrail.
- **SUPPORTED WITH CAVEAT:** The assigned composition conditions produce different
  capped token costs and loop rates at dose 60; this is a degraded-regime result,
  based on three seeds and repeatedly observed evaluation sets.
- **UNSUPPORTED:** Dose 60 demonstrates a usable inference-efficiency improvement;
  even the best condition loses 23.80 accuracy points relative to the pooled base.
- **UNSUPPORTED:** The dose-60 outcome is a null effect; both recorded contrast
  intervals exclude zero and have consistent seed signs.
- **SUPPORTED WITH CAVEAT:** Repetition to the cap occurs outside vLLM as well;
  the correctly loaded HF/PEFT adapter reproduced it on one full-length example.
- **UNVERIFIED:** The exact training mechanism responsible for repetition, or
  complete equivalence between all inference implementations. The diagnostic
  checks are too limited to establish either claim.

Audit limitations: EOS/label verification sampled 48 current targets; backend parity
checks are bounded diagnostics rather than a full rerun. Original generated token IDs
were not saved, so token-count checks validate recorded counts and counting code,
not a reconstruction of every token sequence. No change to the registered analysis or
production pipeline was implemented as part of this audit.

Evidence: `stage1c_dose60_analysis.json`, `stage1c_dose60_independent_audit.json`,
`stage1c_dose60_tokenizer_train.json`, `stage1c_dose60_tokenizer_eval.json`, and
`stage1c_dose60_hf_backend.json`, `stage1c_dose60_hf_backend_full.json`, and
`stage1c_prior_engineering_gate.json` in this directory. Server evidence remains under
`results/stage1c_dose60/independent_audit/` and the earlier `engineering_gate.json`.
