# TMLR strengthening work started — 2026-10-07

Completed:

- Post-hoc cost decompositions, independent-seed bootstrap intervals, paired-seed
  sensitivities, cap sensitivities, and common-finished diagnostics from all existing
  Arm A recipes, per benchmark and pooled (`results/tmlr_existing_analysis/`).
- Missing dataset-construction methods documented from builder/feature code and actual
  build statistics (`docs/tmlr_dataset_methods.md`).
- Frozen evaluation banks: 400 SVAMP and 254 AQuA test problems, with no exclusions
  required by the specified natural-study overlap screen. This does not establish
  absence from model pretraining or exhaustive semantic decontamination.
- Mistral checkpoint revision pinned and all three training subsets checked for
  canonical prefix consistency, supervised terminal EOS, and zero truncation at 4,096.
- Full regression suite: 78 passed. Both branches passed GPU smoke tests.

Running on seeklab-01:

- Qwen: base plus 27 frozen existing checkpoints, evaluated on both new banks.
- Mistral: 18 fixed new trainings (two LRs, three conditions, three seeds), with
  step-15 snapshots where planned, yielding 27 evaluated adapters plus base.
- Each branch automatically verifies raw counts/scoring and saves separate analysis.
- A CPU-only watcher exports combined numerical evidence after both analyses succeed.
  A failure is recorded as incomplete, not silently called a negative scientific result.

Physical GPU allowlist is **0, 1, 2, 3, 4, 7**. GPUs 5 and 6 are prohibited. Worker
leases avoid double allocation between branches; capacity checks wait for busy devices
without terminating other users' processes. All compute is detached from the laptop VPN.

Controller/status paths:
`results/tmlr_strengthening_controller/{qwen,mistral}_STATUS`; combined completion:
`results/tmlr_strengthening_controller/combined_STATUS`.
Final evidence: `results/tmlr_strengthening_evidence/`.

The manuscript is **not yet submission-ready**. External results must be reviewed before
revising claims and conclusions. The updated writing prompt and packet explicitly mark
queued experiments as uncompleted. No journal submission or public release was performed.
