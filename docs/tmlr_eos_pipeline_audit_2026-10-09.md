# Read-only audit of the active terminal-EOS control

Completion addendum: the full matrix subsequently passed all-artifact verification in
`tmlr_eos_final_verification.json`. The scalar-loss interpretation gap was additionally
probed on frozen base/final checkpoints; see `tmlr_eos_effectiveness_probe_summary.md`.
Historical gradient logging remains absent. The running-state audit snapshot and risks
below are preserved as history, not a claim that final verification or the new diagnostic
remain pending.

Date: 2026-10-09. Scope: current v3 EOS implementation, frozen paired configs,
available runtime records, smoke evidence, queue logic, and planned analysis. This is
a main-agent audit, not independent publication-final verification. No training, loss,
coefficient, data, checkpoint, prompt, scorer, or running-job configuration was changed.
The new audit script only reads artifacts and prints evidence.

## Verdict

No confirmed result-corrupting loss, accumulation, EOS-position or pair-configuration
bug was found in the checked paths. That is not a certification of the incomplete
matrix. Full evaluations were not yet available in the runtime audit snapshot.
There were zero completed train+eval jobs and zero recorded failed jobs at that snapshot;
six Qwen high-arm full trainings were active. Do not use this snapshot as live state later.

Evidence: `tmlr_eos_runtime_audit_2026-10-09.json`, the four `tmlr_eos_*_smoke.json`
files, `tmlr_eos_control_precheck.json`, `tmlr_eos_control_scheduler_fix.json`, and
`tests/test_terminal_eos_loss.py` / `tests/test_eos_controller.py`.

## Confirmed findings and limitations

1. **Misleading diagnostic count (low severity, reporting bug).**
   `scripts/train_terminal_eos.py:25` records the return value of full-dataset validation
   as `weighted_terminal_examples`. That is 3676 eligible dataset examples, including
   in a one-step smoke that processes nominally 16 examples. Full60 processes nominally
   60 x 16 = 960 examples, not 3676. This field is not used to compute the loss, stopping
   dose, metrics, or bootstrap; it does not invalidate optimization. Rename/annotate it
   as dataset eligibility, and derive processed counts separately from recorded steps
   and batch/accumulation. Do not rewrite historical records as if they measured the
   processed count, and do not use this field as training-token exposure.

2. **Incomplete cache/provenance dependency guard (medium severity, latent risk).**
   `scripts/evaluate.py:55` fingerprints evaluator source, cfg, data and adapter bytes,
   but not imported prompt/scoring helpers in `src/ila/textio.py`. Changing a helper
   without changing evaluate.py can leave a cache fingerprint unchanged. The final
   analysis requires a nonempty fingerprint but does not recompute it against current
   cfg/data/checkpoint; its raw scoring checks cover some, not all, stale-result paths.
   The precheck also omits some analysis dependencies (external analyzer, bootstrap/
   decomposition helper, loop diagnostic) from its code inventory. Current training
   source hashes match; textio/config hashes are unchanged; the sole observed controller
   hash difference is the explicitly recorded scheduler patch. **No actual stale output
   or unexplained drift was observed.** Before trusting final numbers, independently
   recompute available evaluation fingerprints and preserve all dependency hashes.
   Do not modify the active evaluator now: that would create avoidable code drift.

3. **Intervention-strength diagnostic is missing (scientific limitation, not a proven
   loss bug).** The wrapper records weight, executed loss calls, denominator receipt,
   and eligible terminals, but not native loss versus terminal CE contribution or
   terminal-EOS probability. If teacher-forced EOS is already nearly certain, multiplying
   that token's CE by10 may produce little optimization change. A null result would then
   not rule out broader termination mechanisms. The Qwen smoke's displayed rounded
   total loss is similar across weights, which alone neither proves inactivity nor a
   bug. GPU weight10 records verify 8 loss calls for Qwen and16 for Mistral, with
   accumulated denominators received. Quantify effectiveness after completion using a
   declared fixed diagnostic probe if needed; do not retune the coefficient based on
   outcomes or claim zero intervention merely from rounded logs.

4. **Pooled mechanism gate is not a usable-quality gate (interpretation caveat).**
   `scripts/analyze_tmlr_eos.py:76` checks benchmark-pooled accuracy changes across arms/
   seeds. This implements the frozen operational gate, but gains can offset a bad arm
   or seed. The external analyzer separately computes all-seed baseline-relative
   accuracy/truncation/loop gates. Report those alongside the paired pooled gate;
   `operational_hypothesis_supported` does not mean every condition is quality-safe.
   Mistral's original AQuA boxed-letter baseline limitation remains. Do not change the
   frozen operational gate posthoc or silently substitute the formatting-control prompt.

5. **Zero-cost edge-case analysis failure (low severity, not observed).**
   `scripts/analyze_tmlr_external.py:74` divides by random-arm mean tokens without a
   zero guard. If an arm emits immediate EOS on every problem, relative token effect
   is undefined and the analyzer can raise. Raw absolute effects remain meaningful.
   Handle this as an explicit undefined relative effect if encountered; do not call a
   failed analysis ready, or rerun training solely to avoid this edge case.

## Checks that passed

- All18 fresh matched configuration groups differ only by weight and output roots;
  paired seeds, arms and data SHA256 agree. Runtime configs/data/source hashes for all
  six started full trainings agree with their planned jobs. Allowed GPUs are enforced.
- Weight1 delegates the native Trainer path. Weight10 adds terminal CE with the same
  accumulated-token denominator. Native-value/gradient, split-versus-concatenated, and
  actual Trainer accumulation tests pass. The code rejects smoothing/custom-loss and
  unsupported multi-process paths rather than silently changing normalization.
- Correct causal index: terminal EOS label at position j is predicted by logit j-1.
  Prompt labels and internal EOS tokens are not newly weighted. Qwen's serialized
  whitespace suffix (token198) remains weight1; Mistral has no such suffix. The GPU
  weight10 smokes record EOS151645/2, the expected suffix sets, proper accumulated
  normalization, and actual adapter evaluation—not base-model fallback.
- The queue's corrected wait loop exits when all jobs are claimed. Parent still waits
  for every worker and rejects failures/incomplete counts before analysis. The separate
  scheduler hash amendment is recorded. No missing result is converted into success.
- Paired analysis aligns all seeds/UIDs/gold, uses equal benchmark/arm/seed weights and
  shared question resampling. Seed pairing within each arm is preserved across the two
  benchmarks. Finished-cohort decomposition remains explicitly descriptive.
- Full server regression suite: **94 passed** again during this audit. No new GPU
  experiment or parameter search was launched for the audit.

## Recommendation

Let the current runs continue. The identified count-label and guard weaknesses do not
justify restarting expensive training on the evidence available. Before treating the
matrix as paper evidence, audit final adapter/data/config fingerprints, recompute raw
metrics, distinguish pooled mechanism support from per-seed usable quality, and qualify
null conclusions by the missing intervention-strength diagnostic. Correct the metadata
label and strengthen dependency hashing in a separately recorded maintenance change;
do not silently alter an active experiment. Rerun only if a substantive mismatch or
optimization/scoring bug is actually demonstrated.
