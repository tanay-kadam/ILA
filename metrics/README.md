# Archived numerical metrics

This archive preserves the selected numerical results from the SSH project as of
2026-10-20: 611 evaluation summaries, 381 run records, 208 structured training
logs, 42 terminal-EOS records, aggregate analyses, quality gates, and two metric
CSVs. The 1,286 entries include smoke tests, failed/interrupted work, historical
recipes, and final experiments. Their presence is not evidence that a run passed
its scientific or engineering gates. Use the existing claim guidance in `docs/`.

`manifest.json` maps original server paths to published repository paths and
records exact byte counts and SHA-256 hashes. Of these entries, 19 were already
published under `docs/`; the other 1,267 are under `metrics/results/`, retaining
the original relative hierarchy. Files are copied unchanged. The checksum
manifest preserves provenance; it does not independently validate the scores.

Use the existing analysis scripts with the appropriate archived directory, or
restore files to their original paths using the manifest if a script expects
the original layout. Relative references inside archived JSON remain historical
paths; raw predictions and trained weights referenced there are not included.

Raw datasets, generated answers, bulk text logs, tokenizer assets, model weights,
and environment directories are outside this archive. Raw-to-metric replication
requires the original raw artifacts. Numerical summaries alone do not provide it.

Historical server source differences are preserved in
`../archive/server_source_20261020/`; maintained scripts remain in `../scripts/`.
