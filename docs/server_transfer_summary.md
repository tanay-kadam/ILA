# SSH repository transfer — 2026-10-07

Source: `tkadam@seeklab-01.ece.ncsu.edu:/home/tkadam/ILA`
Destination: `C:/Users/kadam/ILA`

Transferred and checksum-verified 2,521 missing or changed non-weight files
(2,882,917,172 bytes, approximately 2.68 GiB). This includes training/evaluation
results, raw generations, analyses, run records, logs, data, configs, source code,
tests, and documentation. Files already matching the server were not downloaded.

Virtual environments, dependency caches, Python bytecode, pytest temporary
directories, Git internals, and transfer staging were excluded. By user choice,
261 model/checkpoint weight files remain on the server. Their exact paths and sizes
are recorded in `.transfer/transfer_report.json`.

Six differing local files were preserved. Their server versions are available under
`.transfer/remote_conflicts/` at the corresponding relative paths:

- `.gitignore`
- `requirements.txt`
- `configs/stage1_distill.yaml`
- `scripts/setup_vllm_env.sh`
- `tests/test_paraphrase.py`
- `tests/test_styles.py`

One Linux filename contains a carriage return unsupported by Windows:
`results/stage1_pilot1_4000tok/eval_armA_random_seed0.flag` followed by CR.
Its local name is `eval_armA_random_seed0.flag%0D`; the reversible mapping is
recorded in the transfer report. No file content was changed to resolve this.

The compressed transport archive and checksum manifest remain in `.transfer/`.
The archive checksum and every selected file's checksum were verified during
installation. Original server artifacts and local-only files were retained.

Model configuration/index files may reference weights that intentionally remain
remote. Local analysis of raw outputs is available; local checkpoint inference
requires those weight files separately.
