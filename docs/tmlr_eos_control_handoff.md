# Terminal-EOS experiment handoff (2026-10-09)

Completion update: all36 jobs and72 evaluations completed, zero failures, automatic
analysis passed at07:53:46 server local time on2026-10-09. Status:
`COMPLETED_VERIFIED_EOS_CONTROL_ANALYSIS_READY`. Summaries are now local in
`tmlr_eos_control_comparison.json` and four `tmlr_eos_*_weight*_analysis.json` files.
Both families failed the frozen operational cap-reduction test. The launch-state
description below is historical, not live status. Independent final verification is
separate from the completed automated checks.

Scope: experiments only. Manuscripts and historical primary scores are unchanged.
Current protocol: `tmlr_eos_control_protocol.md`. Current ChatGPT evidence lives in
`experiment_updates.json`, generated from `experiment_registry.json`.

Matrix: 36 fresh trainings = two model families x weights1/10 x high/low/random x
seeds0/1/2. Each stops at step60 of the original planned230 cosine schedule at LR1e-5.
Final adapters receive 400 SVAMP and 254 AQuA evaluations with the original fixed
prompts, greedy decoding, cap6144, and unchanged primary scorers: 72 full evaluations.
Fresh weight1 controls avoid relying on historical Qwen records without modern hashes.
Only terminal EOS is reweighted; serialized whitespace after it remains supervised
at its original weight. Internal EOS tokens are not upweighted. Save adapters only.

Verified checks: native loss/gradient and actual Trainer accumulation tests, suffix
position tests, and 94 server regression tests passed. All eight new loss/Trainer tests
are included in that total. Full-benchmark outcomes do not yet exist in this handoff.

Current server namespace: `/home/tkadam/ILA/results/tmlr_eos_control_v3`.
Verified status at 2026-10-09 02:11:53 server local time:
`SMOKES_PASSED_RUNNING_36_TRAIN_AND_72_EVAL_QUEUE`. All four GPU train+adapter-eval
smokes passed; full jobs started. They are not completed in this handoff.
Detached controller launch PID: 2110558 (verify command before using PID; it may later
be reused). Launch record: `results/tmlr_eos_launch_v3.json`.
Status: `results/tmlr_eos_control_v3/controller/STATUS`.
Log: `results/tmlr_eos_control_v3_launch.log`.
GPU allowlist: 0,1,2,3,4,7; never 5 or 6. Shared leases and capacity checks prevent
our branches from overlapping; no other user's job is killed/preempted.

The controller runs precheck -> weight1/10 train+adapter-eval smokes per family ->
36 train/eval jobs -> full artifact checks -> seed-resolved analyses -> paired comparison.
Expected successful final status: `COMPLETED_VERIFIED_EOS_CONTROL_ANALYSIS_READY`.
The paired report is `controller/comparison.json`; each family/weight also has
`weight{1,10}/eval/analysis.json`. Failures retain logs and halt the analysis-ready claim.
PC sleep or VPN disconnection does not terminate the detached server session.

Pre-full scheduler fix: `controller/scheduler_fix.json` preserves original precheck and
updated worker-controller hashes. Only waiting-worker exit behavior changed; scientific
configs/loss remain unchanged. The old interrupted controller's proven dead GPU0 lease
was reclaimed; no other process's lease or result data was removed.

Retained startup attempts, neither a completed scientific matrix:

- V1 `results/tmlr_eos_control`: CPU precheck failed on assuming EOS was the final label.
  Qwen's canonical serialization places whitespace after EOS. No training started.
- V2 `results/tmlr_eos_control_v2`: stopped during smokes, before full runs, to disable
  inherited merged-model saving and enforce a nonempty trained adapter. Its weight1
  evaluation smoke cannot validate adapter loading and must not be treated as evidence.
  Its sources and generated artifacts are retained; nothing material was deleted.
- V3 fixes both safeguards in a new namespace, pins/hashes cached model snapshots,
  checks all data/terminal positions and >=90 GiB free before training, and requires
  actual adapter paths. These are new-control engineering fixes, not demonstrated bugs
  in the original scientific results.

Do not rerun `launch_tmlr_eos.py` over existing results or use `--force`. Inspect any
failure and document a separate versioned retry. Results are saved remotely; collection
and packet refresh occur when checked/pulled, not by a background laptop monitor.

Interpretation: improved termination under the intervention would support sensitivity
to EOS supervision, not sole-cause identification or causal reflection. A negative or
model-dependent result is retained. Trace question/length/dose matching and blinded
human validation remain not started.
