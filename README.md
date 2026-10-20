# ILA — Inference Liability Attribution

## Current research artifact (October 2026)

## Recovery from GitHub

**Status (2026-10-20): scripts, configs, and selected numerical metrics are
preserved in GitHub. The private Hugging Face backup was never completed. The
owner intentionally deleted the SSH project's trained adapters and merged model
weights: 361 files totaling 368,807,418,240 bytes. These trained weights cannot be
restored from this repository or the proposed Hugging Face backup.**

### 1. Recover source and numerical results

Install Git and Python. On a new machine:

```bash
git clone https://github.com/tanay-kadam/ILA.git
cd ILA
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install huggingface_hub numpy pyyaml pytest
```

This is an inspection environment, not the full GPU training stack. Maintained
code is in `scripts/` and `src/`; configurations and tests are also included.
Consolidated analyses and scientific limitations are under `docs/`.

`metrics/manifest.json` maps 1,286 original numerical metric/provenance files to
published paths, including 19 files already under `docs/` and 1,267 additional
files under `metrics/results/`. It includes per-run evaluation summaries,
structured training logs, run records, and aggregate analyses. See
`metrics/README.md` for scope. Exact hashes and byte counts permit verification.
Relative paths inside the archived records remain historical references.

`archive/server_source_20261020/manifest.json` records the server source snapshot;
ten older differing source/config/test versions are preserved separately. Keep
these provenance copies separate from maintained source.

### 2. Hugging Face and trained-weight availability

No private backup revision or downloadable adapter archive exists for this
project. The former `tk-8503/ila-private-backup` plan was not completed. The
backup/restore scripts remain as historical utilities; their presence does not
mean any checkpoint is available. A trained adapter cannot be reconstructed
from a metric JSON file. Retraining requires the original data, recipe, and
suitable GPU resources, and does not guarantee bit-identical weights.

Base models may be obtained separately from their original Hugging Face model
providers under their licenses. Use the recorded model identifiers/revisions;
base-model downloads do not recover the deleted trained adapters or merged models.

### 3. GPU setup and reproduction boundaries

Use a Linux NVIDIA GPU machine for training/evaluation. Consult
`docs/tmlr_train_environment_versions.json`, `docs/tmlr_eval_environment_versions.json`,
run-specific configs, and `docs/tmlr_reproducibility_manifest_2026-10-09.json`.
Do not blindly upgrade historical libraries or assume `requirements.txt` pins
the original environment. Training and vLLM evaluation may require separate
environments. Older records describing server weights predate their deletion;
see `docs/server_weight_deletion_2026-10-20.json` for the deletion receipt.

### Before deleting any remaining local or SSH files

This GitHub snapshot does not contain raw dataset caches, bulk predictions/logs,
local-only PDFs, environment directories, or `.transfer/`. Preserve any such
files you need separately. The SSH nonweight files were kept and verified
unchanged after weight deletion; their continued presence is not a GitHub backup.
Numerical summaries permit inspection but are not a complete raw-to-metric
replication package. No further deletion is performed by these instructions.

## Research summary

This repository preserves an exploratory study of training-trace composition,
capped generation cost, answer quality, and termination under LoRA adaptation.
It does **not** establish a causal effect of reflection or a successful inference-cost
attribution method. The original proposal and early-stage descriptions below are
historical, not claims of a passed quality gate.

Start with `docs/experiment_updates.json` for the consolidated evidence and
`docs/paper_evidence_claim_map_2026-10-09.md` for claim boundaries. The external
Qwen/Mistral analyses, fixed-prompt control, fresh EOS control, and diagnostics
are under `docs/`; executable research code is under `scripts/` and `src/`.
AI-generated qualitative labels are not human validation.

The public source snapshot excludes model weights, virtual environments, credentials,
raw dataset caches, and bulk prediction/log archives. Archived numerical analyses
are included; this is not a complete independent reproduction of every historical run.
No paper acceptance, submission, or review status is asserted. This identified GitHub
repository is not an anonymized double-blind supplement. No redistribution license
for third-party data or models is granted by this repository.

Research code for attributing LLM deployment cost to training data. See
`Inference_Liability_Attribution_Research_Proposal.pdf` for the full proposal.

Current stage: **Gate 0** (does training-data reasoning style change inference cost on
unseen queries?) with a preliminary **Gate 1** test (does reasoning *structure* matter at
matched training length?).

## Gate 0 experiment design (fixed before running)

**Hypotheses.**
- G0: SFT on a reasoning style changes mean generated tokens on unseen GSM8K test
  questions relative to SFT on concise traces.
- G1: At matched per-example training length, styles that revisit earlier work
  (`verify`, `backtrack`, `resolve`) produce different inference cost than a linear,
  forward-only elaboration (`detailed`).

**Falsification / decision rules** (`configs/gate0.yaml`, `analysis:`; applied by
`scripts/analyze_gate0.py` on GSM8K test, 3 seeds required):
- G0 PASS if some style vs `concise` has a hierarchical-bootstrap 95% CI excluding 0,
  the same sign in every seed, and |delta| >= 3x the between-seed SD of `concise`.
  Otherwise FAIL => stop the project.
- G1 PASS if some revisiting style vs `detailed` is significant (CI excludes 0,
  seed-consistent) with |relative delta| >= 10%. FAIL if every style is non-significant
  or < 5%, meaning liability at this level is explained by length => stop or reframe.
  Anything in between is INCONCLUSIVE => add seeds before deciding.

**Independent variable.** Only the style of the SFT target text. Questions (3,000 GSM8K
train problems, sampled with seed 0), final answer line, prompt format, hyperparameters,
seeds, and evaluation code are identical across styles.

**Styles** (generated programmatically from GSM8K calculator annotations in
`src/ila/styles.py`; every stated equation is arithmetically true, except deliberately
abandoned false starts in `backtrack`):

| style | structure |
|---|---|
| concise | original GSM8K steps |
| detailed | step labels, restated givens, place-value breakdowns; never revisits |
| verify | inverse-operation checks after steps, then repeated re-checks |
| backtrack | an erroneous attempt before steps, explicitly abandoned |
| resolve | full redundant recomputation after the solution |

Each non-concise style has a ladder of insertion levels. Per problem, one level per style
is chosen so that all four are within 10% token length of a common target of at least
1.6x the concise length. Built data: 7,109 of 7,473 problems matchable. Mean target
tokens are concise 92.6, detailed 195.5, verify 196.4, backtrack 195.8, resolve 194.8,
with a mean max per-problem deviation of 4.3% (`data/gate0/build_report.json`).

**Model and training.** Qwen2.5-0.5B *base* (no instruct prior on output style). LoRA
r=16 on all linear projections, lr 2e-4, 2 epochs, effective batch 16, completion-only
loss, EOS appended. Final checkpoint only; no checkpoint selection on any eval data.

**Evaluation.** Greedy decoding. GSM8K test (all 1,319 questions, unseen and never
filtered; 1,024-token cap) is primary. MATH-500 (2,048-token cap) is a secondary
out-of-distribution cost measurement whose accuracy uses approximate string matching.

**Metrics.** Per question: generated tokens excluding EOS (truncated outputs counted at
the cap and reported as a truncation rate), correctness, and style-marker presence.
Aggregates include mean tokens, accuracy, tokens per correct answer (total tokens / number
correct), and *amplification*: generated tokens divided by the length of a faithful
imitation of the style on the same test question. Amplification > 1 means the model
spends more than it was taught to.

**Seeds and uncertainty.** 3 seeds (control LoRA init and data order). Contrasts
report a hierarchical bootstrap (prompts and seeds resampled) and per-seed deltas.

**Leakage checks.** GSM8K train/test are disjoint; the builder asserts no exact
question overlap. The eval set is never filtered by the style builder.

**Known limitations / alternative explanations to check before trusting a result.**
- Templated phrasing: styles differ in surface markers, so the effect may reflect
  template imitation. Marker rates in outputs measure how much style transferred.
- `detailed` includes arithmetic breakdowns that may change accuracy; compare tokens
  per correct answer, not only raw tokens.
- Truncation makes mean tokens a lower bound; inspect truncation rates by style.
- One model, one domain, LoRA only: Gate 0 is a go/no-go check, not a paper result.

## Stage 1: natural-data gate (fixed before running)

Gate 0/1 found that fixed template wording, not reasoning structure, drives over-generation
on synthetic traces. Stage 1 asks whether the project can support a main-track paper on
real data. It decides two things, with no attribution method involved:

1. Does *which* training traces are included change deployed generated-token cost by more
   than training-seed noise?
2. How much of that effect do cheap trace heuristics (mean length, reflection density,
   restart count, answer repetitions, source mix) explain?

**Data.** `open-r1/OpenR1-Math-220k`, `default` split. One verified-correct, `stop`-finished
R1 trace per problem, 1,000 to 4,000 Qwen tokens. Problems with a 13-gram overlap (or exact
match for short problems) with MATH-500 or GSM8K test are removed. Pool target 20,000.
Per-trace features (`src/ila/features.py`): token length, reflection density per 1k tokens,
restart count, `\boxed{}` repetitions, source, problem type.

**Arm A (9 runs, seeds 0 to 2).** Pool bins = length decile x source x problem type.
Within each bin of at least 6 traces, `high_refl` takes the top third by reflection density
and `low_refl` the bottom third; `random` takes the same per-bin counts at random from the
whole bin. All three sets have 4,000 traces and identical bin composition by construction.
Report-only evidence (not a stopping rule): is reflection density causal at matched length?

**Arm B (20 runs).** TF-IDF k-means (k=16) on problem text. 16 subsets of 4,000 traces with
cluster weights ~ Dirichlet(0.5), capped at cluster size with the excess redistributed.
One seed each (seed 0), plus seeds 1 and 2 on subsets `b00` and `b01` for noise.

**Model and evaluation.** Qwen2.5-1.5B-Instruct, LoRA r=64, 2 epochs, max length 4,608,
completion-only loss, plain-text prompt (no chat template) with the R1 boxed instruction.
Greedy decoding, 6,144-token cap. Eval sets: all 500 MATH-500 questions and a fixed
500-question GSM8K-test subset. Cost = generated tokens (truncated outputs at the cap).

**Analysis (`scripts/analyze_stage1.py`).**
- Outcome per run = mean generated tokens over all 1,000 eval questions (primary).
- Seed-noise variance = pooled within-group seed variance over every (set, seed) group with
  3 seeds (3 Arm A sets + `b00`, `b01`), df = 10.
- Between-subset variance = variance of the 16 Arm B seed-0 means.
- Explainable variance = between-subset variance minus noise variance; significance by a
  one-sided F test (15, 10 df) at 0.05. Explainable fraction f = explainable / between.
- Heuristic R2 = leave-one-subset-out R2 of ridge regression on standardized subset-level
  features (alpha chosen from a fixed grid by LOO error, which favors the heuristics).
- Heuristic share = max(0, R2) / f, capped at 1. Prompt-bootstrap CIs are reported.
- Cost range = (max - min) / mean of the 16 Arm B means.

**Pre-registered decision rules (applied to point estimates; do not edit after seeing results).**
- STOP, no effect: F test not significant, or cost range under 5% of the mean.
- STOP or pivot, heuristics suffice: heuristic share >= 0.80.
- GO to Stage 2: F test significant, cost range >= 5%, heuristic share <= 0.60.
- AMBIGUOUS: share in (0.60, 0.80). Discuss; likely add 16 more subsets.

**Pilot gate (before Arm A / Arm B).** Run `random` seed 0 first. Continue only if
truncation < 20% on both eval sets and MATH-500 accuracy clearly exceeds the base model.
Otherwise shorten traces or raise caps and re-pilot once; if it still fails, the 1.5B scale
is wrong for this question.

**Pilot 1 result (traces 1,000 to 4,000 tokens, max length 4,608): FAIL.** `armA_random`
seed 0 vs the untuned Qwen2.5-1.5B-Instruct (same prompt, same decoding):

| set | base acc / tokens / trunc | pilot acc / tokens / trunc |
|---|---|---|
| GSM8K subset | 0.688 / 320 / 0.8% | 0.420 / 2878 / 40.6% |
| MATH-500 | 0.504 / 827 / 5.2% | 0.148 / 4831 / 75.2% |

Inspection of the GSM8K outputs: all 203 truncated outputs end in degenerate repetition
(e.g. endless numbered lists), not long legitimate reasoning, so raising the cap cannot
help. Non-truncated outputs have median 612 tokens and 68% accuracy, close to the base
model. Applying the pre-registered fallback: traces limited to 2,500 tokens (max length
3,072), data and pool rebuilt, one re-pilot. Pilot 1 artifacts are kept in
`results/stage1_pilot1_4000tok` and `data/stage1_v1_4000tok`. If the re-pilot also fails,
the rule above applies: discuss moving to a larger model before spending more compute.

**Pilot 2 result (traces 1,000 to 2,500 tokens, max length 3,072): FAIL.** GSM8K subset
accuracy 0.436, 2,669 mean tokens, 36.4% truncated; MATH-500 accuracy 0.150, 4,491 mean
tokens, 68.6% truncated (base: 0.688 / 0.504 accuracy, 0.8% / 5.2% truncation). Shortening
traces barely changed the outcome. Per the pilot rule no Arm A / Arm B runs were launched;
the decision on scale or measurement design is open.

**Stage 1b measurement repair (registered 2026-10-04).** Inspection of the Qwen2.5-7B
plain-prompt baseline showed that capped outputs generally answered within a few hundred
tokens and then entered severe repetition loops. A 50-question-per-set diagnostic using
the model's canonical chat template eliminated truncation. The prospective revised
baseline and one-pilot gates are fixed in `docs/stage1b_measurement_revision.md`. The
original Stage 1 results remain failures under the original protocol.

Files: `configs/stage1.yaml`, `src/ila/features.py`, `src/ila/natural.py`,
`scripts/build_natural.py`, `scripts/analyze_stage1.py`, `scripts/launch_stage1.sh`.

## Layout

```
configs/gate0.yaml        experiment config (thresholds are pre-registered)
configs/smoke.yaml        CPU smoke test with a tiny random model
src/ila/                  parsing, style generation, prompt/answer utilities
scripts/build_data.py     build SFT and eval sets
scripts/train_sft.py      LoRA/full SFT for one (style, seed)
scripts/evaluate.py       greedy decoding and cost metrics (vLLM or HF)
scripts/analyze_gate0.py  aggregation, bootstrap, gate decisions
scripts/run_gate0.sh      whole matrix, resumable
tests/                    unit tests for style generation
```

## Running on the GPU machine

```bash
python -m venv .venv && source .venv/bin/activate
pip install vllm && pip install -r requirements.txt
python -m pytest -q tests

# optional: 1-run smoke on GPU before the full matrix
python scripts/build_data.py --config configs/gate0.yaml
python scripts/train_sft.py --config configs/gate0.yaml --style verify --seed 0 --set train.max_steps=20 results_dir=results/gpu_smoke
python scripts/evaluate.py  --config configs/gate0.yaml --style verify --seed 0 --eval-set gsm8k_test --limit 50 --set results_dir=results/gpu_smoke

# full matrix: 5 styles x 3 seeds, then analysis
bash scripts/run_gate0.sh configs/gate0.yaml
cat results/gate0/analysis.md
```

Every run writes `run_record.json` (command, resolved config, device, versions, status)
and refuses to fall back silently to CPU. Completed runs are skipped on re-run;
partially completed runs stop with an error until inspected or `--force` is given.
