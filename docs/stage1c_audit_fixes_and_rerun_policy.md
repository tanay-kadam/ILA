# Audit fixes and rerun policy (2026-10-07)

The independently checked 15-step matrix and nine dose-60 vLLM runs do not require
retraining or regenerated predictions. The discovered HF adapter bug was outside
their executed backend. Reporting and guardrail changes are handled by a separate
`analysis_v2` report; the original analysis and predictions remain preserved.

Implemented fixes:

- HF evaluation explicitly loads the recorded PEFT adapter and base tokenizer.
- Evaluation caches require matching config, evaluator, data, and adapter fingerprints.
  Unverified legacy caches are preserved and refused for automatic reuse.
- New generations save token IDs and backend/library versions.
- Training records save code/data hashes and physical GPU selection; completed runs
  refuse mismatched configs or changed recorded training data.
- Composition signals and quality failures have distinct outcome labels.
- New guardrails additionally check base-relative accuracy and truncation for each
  benchmark and training seed. These added checks are retrospective diagnostics for
  old runs, not amendments to their original preregistration.
- Regression tests cover the HF adapter-loading route, cache invalidation,
  significant-but-degraded outcomes, baseline quality failures, and gate boundaries.
  All 70 repository tests passed on the server before launch.

New training is required to test the revised learning rate. The final attempt changes
only LR to 3e-6, with dose 60 on the planned 230-step schedule. Seed-0 engineering
pilots gate the conditional three-seed comparison. The maximum is nine new trainings
and one one-step smoke test, rather than a repeat of the 29-run Arm A/B matrix.
The experiment plan is `stage1c_final_stability.md`.

Any historical adapter-only HF evaluation would need corrected re-evaluation if used
as evidence. This is a conditional policy, not a finding that the audited vLLM results
were generated through that branch.

Original server scripts were backed up under `results/stage1c_audit_fix_backup/`.
The old dose-60 analysis revision is `results/stage1c_dose60/analysis_v2.json`;
the new experiment ends with `results/stage1c_final_stability/final_report.json`.
