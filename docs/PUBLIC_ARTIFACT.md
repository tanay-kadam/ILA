# Public research artifact

Repository: https://github.com/tanay-kadam/ILA

## Evidence entry points

- `experiment_updates.json`: consolidated current and historical numerical evidence,
  protocols, experiment registry, uncertainty, and interpretation guardrails.
- `paper_evidence_claim_map_2026-10-09.md`: supported claims and limitations.
- `tmlr_external_qwen_analysis.json` and `tmlr_external_mistral_analysis.json`:
  external benchmark outcomes, base results, all recipes, and per-seed statistics.
- `tmlr_format_control_comparison.json`: fixed AQuA prompt-package control.
- `tmlr_eos_control_comparison.json`: fresh matched EOS-weight comparison.
- `tmlr_eos_final_verification.json`: recorded pipeline integrity checks.

Executable evaluation, analysis, loop-detection, and training code is in `../scripts/`
and `../src/`; run configurations are in `../configs/`. In particular, the loop
helper is `scripts/diagnose_stage1_outputs.py::loop_metrics`. An available helper
is not the same as a human-validated annotation procedure.

## Verification and scope

On 2026-10-10, the local suite passed 104 tests with the project `src` on the
Python import path and a fresh temporary directory. This is a software check,
not an independent reproduction of the scientific experiments.

SSH comparison found that local source versions included later corrections not
present on the server (including full-trace identity matching and portable-path
handling). They were retained, not overwritten by older remote versions. Remote
differences are staged locally outside Git for provenance review.

The identified public repository excludes model weights, virtual environments,
credentials, raw dataset caches, bulk raw predictions, logs, and local transfer
archives. Bulk nonweight SSH synchronization is a separate local backup process;
its completion must not be inferred from publication of this source snapshot.
Numerical summaries permit inspection and table reconstruction but are not a
complete raw-to-metric replication package. Dataset/model rights remain with
their respective owners. No third-party redistribution license is implied.

On 2026-10-20, `../metrics/manifest.json` was added to preserve 1,286 selected
numerical metric/provenance files: 19 reference existing published files and
1,267 are additional exact-byte copies. Historical server source versions are
mapped in `../archive/server_source_20261020/manifest.json`, with ten differences
preserved separately. These archives do not revise scientific scores or claims.

The owner intentionally deleted 361 trained adapter/merged-model weight files
from the SSH project (368,807,418,240 bytes). All 5,327 inventoried nonweight files
were checksum-verified unchanged afterward; see
`server_weight_deletion_2026-10-20.json`. No private Hugging Face weight backup
was completed. Historical documents that describe weights as remaining on the
server predate this deletion. Shared base-model caches were outside its scope.

No human validation of the loop heuristic has been completed. AI-only annotation
agreement must not be reported as human reliability or annotation accuracy.

For a paper, cite a specific Git commit rather than a moving branch. This personal
GitHub URL reveals author identity and must not be represented as anonymized.
