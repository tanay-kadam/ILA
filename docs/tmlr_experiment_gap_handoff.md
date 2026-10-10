# Experiment-only handoff, 2026-10-08

The manuscripts, writing prompt, and primary evidence packet were not changed in this
turn. Original predictions and scores are preserved. The goal is stronger experimental
validity, not a favorable outcome or a guaranteed acceptance.

## Completed now

- Gold-blind boxed-option-content audit of all 28 AQuA checkpoints per family.
  Local evidence: `tmlr_format_audit_qwen.json` and `tmlr_format_audit_mistral.json`.
  Deterministic cases: corresponding `.cases.jsonl` files; **not human reviewed**.
- Mistral base: 27/254 strict-correct; 28 additional uniquely mapped content-correct
  answers. Secondary diagnostic is 55/254 (21.65%), versus 10.63% primary accuracy.
  Qwen base: 213/254 under both rules; no recovered correct answers.
- Remote regression suite: 83 tests passed; local suite: 91 tests passed. New mapper
  and prompt tests passed. Local Windows temp-directory sandbox permissions initially
  blocked seven fixture setups; a fresh authorized test directory resolved that issue.
- Frozen inventory, resolved configs, adapter/data/code hashes, and original/control
  input-token counts are recorded in `tmlr_format_control_precheck.json`.
- Both base and adapter GPU smoke tests passed for both families on physical GPU 0.
  Smoke results are intentionally not evidence of full-benchmark performance.

## Completed follow-up: fixed prompt-format control

The fixed AQuA formatting-prompt control has 56 full checkpoint evaluations: both bases
and all three recipes x three arms x three training seeds per family. No new training
or checkpoint selection occurs. See `tmlr_format_control_protocol.md` for the frozen
hypothesis, falsification threshold, controls, and limitations.

Server: `tkadam@seeklab-01.ece.ncsu.edu`, repository `/home/tkadam/ILA`.
Detached controller PID at launch: **1292489** (PID may be reused after completion).
Verified controller status: `COMPLETED_VERIFIED_FORMAT_CONTROL_ANALYSIS_READY`,
completed 2026-10-08 at 14:05:19 server local time: 56 jobs completed, zero failed.
Worker allowlist: **0,1,2,3,4,7**. GPUs 5 and 6 are excluded, not merely deprioritized.
Workers share GPU leases and wait for capacity instead of preempting other users.
PC sleep/VPN disconnect should not terminate the detached server controller.

Paths on server:

- `results/tmlr_format_control_launch.log`
- `results/tmlr_format_control_controller/STATUS`
- `results/tmlr_format_control_controller/{qwen,mistral}_queue/`
- `results/tmlr_format_control_{qwen,mistral}/`

The controller automatically verifies all expected jobs, independently recomputes
metrics, runs the secondary content audit, and writes per-family `analysis.json` plus
`results/tmlr_format_control_controller/comparison.json`. A successful terminal status is
`COMPLETED_VERIFIED_FORMAT_CONTROL_ANALYSIS_READY`. Any failures retain their logs and
stop the analysis-ready claim. Analysis/comparison and automated audit JSON files have
now been pulled into `docs/`. Raw predictions and weights remain remote; there is no
automatic laptop download or continued Codex monitoring.

Do not restart the controller or use `--force` over the existing roots. It deliberately
refuses an existing controller directory. Inspect retained failures before a separate,
explicitly versioned retry. Original studies must not be rerun because of this audit.

## Next experimental decisions

First inspect the fixed prompt control, including reversed/null effects, and whether
Mistral's format validity improves without hiding cap hits or repetition. This closes
one accuracy-interpretation threat; it does not identify the cause of adaptation failure.

Highest-value remaining work:

1. A matched terminal-EOS supervision intervention with exact native-loss/gradient
   equivalence tests, to probe termination behavior rather than infer it from a
   descriptive cost decomposition. Separate fixed protocol required before training.
2. Question-, length-, and training-dose-matched composition controls if the intended
   claim is causal about reflection; otherwise retain the composition-associated claim.
3. An actual decoding-cap intervention on fixed representative conditions. Clipping
   old generated lengths is not an experiment in longer-run termination.
4. Blinded human review of repetition and answer-format cases. Exported samples are
   not validated labels; automated scoring cannot supply independent human review.

Model-family replication and external benchmarks are already complete; broad extra
model/LR searches are not the immediate priority. New controls can weaken the story,
which is scientifically useful and must be reported honestly.

## Current ChatGPT update handoff

Use `CHATGPT_EXPERIMENT_UPDATE_PROMPT.md` with `experiment_updates.json` and the current
manuscript sources. The self-contained packet supersedes old queued-extension prose,
not preserved historical numbers. Maintain `experiment_registry.json` and regenerate
the packet after each experiment update. Proposed termination/matching/human-validation
controls remain not started; none is automatically queued by the packet generator.
