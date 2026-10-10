# Stage 1 session summary and recommendation

**Date of this note:** 2026-10-04  
**Project:** Inference Liability Attribution (ILA) / TraCIL  
**Goal at start of this work:** A publishable main-track paper. Stage 1 was the pre-registered natural-data gate meant to decide whether that paper can rest on attributing *generated-token cost* to training-trace composition.

This document summarizes the Stage 1 implementation and all experiments run in this conversation, then records the recommended next step in detail.

> **2026-10-06 revision:** The later 15-step Qwen2.5-7B natural-composition matrix
> completed and passed an independent artifact audit. Its registered outcome was
> `STOP_NO_EFFECT` (Arm B F=0.583, one-sided p=0.833). This is retained as a
> low-dose null—not a claim that composition can never matter. The next registered
> experiment is the compute-capped dose-60 Arm A follow-up in
> `docs/stage1c_dose_response.md`: high/low/random with three seeds, seven new
> trainings and two reused trajectory checkpoints. Step 230 is locked behind a
> prospective signal-and-guardrail gate and cannot launch automatically.

---

## 1. Project context (what we already knew)

### 1.1 Scientific question

ILA asks whether one can attribute LLM *inference cost* (generated tokens at deployment) to *training data*. The proposed estimator is TraCIL. Before TraCIL itself, Stage 1 asks a simpler gate:

> Does training-trace composition change deployed generated-token cost beyond seed noise, and how much of that change is explained by cheap heuristics (length, reflection density, restart count, boxed repeats, source mix)?

### 1.2 Prior gates (Gate 0 / Gate 1)

Earlier work on **synthetic GSM8K styles** with Qwen2.5-0.5B / 1.5B found that **fixed template wording**, not reasoning structure, drives about **7–17% over-generation**. That positive result is still the strongest empirical finding in the project.

### 1.3 Stage 1 design (pre-registered)

Documented in `README.md` and `configs/stage1.yaml`. Summary:

| Piece | Specification |
|---|---|
| Model (initial) | Qwen2.5-1.5B-Instruct, LoRA r=64, α=128, lr 1e-4, 2 epochs |
| Data | OpenR1-Math-220k; one verified-correct, stop-finished trace per problem; 13-gram decontamination vs MATH-500 and GSM8K test |
| Arm A | `armA_high` / `armA_low` / `armA_random` × seeds 0–2 (9 runs); matched bins = length decile × source × problem type; high/low take extremes of reflection density |
| Arm B | 16 Dirichlet cluster-mixture subsets (TF-IDF k-means), seed 0, plus seeds 1–2 on `b00`,`b01` (20 runs) |
| Eval | GSM8K 500-subset + MATH-500; greedy; `max_new_tokens` 6144; plain prompt with `\boxed{}` answer format |
| Analysis | Pooled seed noise; one-sided F test; leave-one-out ridge R² on heuristics; pre-registered STOP / GO / AMBIGUOUS rules |
| Pilot gate | Truncation &lt; 20% on **both** eval sets **and** MATH-500 accuracy above the untuned base. Fail once → shorten traces to 2,500 tokens or raise cap and re-pilot once. Fail again → “1.5B scale is wrong; discuss moving to 7B before spending more.” |

Decision rules (fixed before seeing results):

- **STOP_NO_EFFECT** — F test not significant, or cost range &lt; 5%
- **STOP_HEURISTICS_SUFFICE** — heuristic share ≥ 0.80
- **GO_STAGE2** — significant, range ≥ 5%, share ≤ 0.60
- **AMBIGUOUS** — share in (0.60, 0.80)

---

## 2. What was implemented in this session

All Stage 1 code and analysis were built and tested. Plan to-dos were completed except the 29-run matrix, which was cancelled after pilot failure.

### 2.1 Code and configs

| Artifact | Role |
|---|---|
| `configs/stage1.yaml` | Standalone Stage 1 config (later fallback: `max_tokens` 2500, `max_len` 3072) |
| `src/ila/features.py` | Reflection / restart / boxed features |
| `src/ila/natural.py` | Decontamination, matched Arm A, TF-IDF clustering, Dirichlet Arm B |
| `src/ila/textio.py` | Prompt builder; brace-matched `\boxed{}` extraction |
| `scripts/build_natural.py` | Build pool, Arm A/B train files, eval sets, leakage report |
| `scripts/analyze_stage1.py` | Noise variance, LOO ridge, decision rules, `--pilot` |
| `scripts/evaluate.py` | HF + vLLM backends; `--base`; stop on eos/pad |
| `scripts/train_sft.py` | LoRA SFT; completion-only; gradient checkpointing; chat-free prompt |
| `scripts/launch_stage1.sh` / `queue_worker.sh` | Detached multi-GPU queue with atomic job claims |
| `scripts/setup_vllm_env.sh` | Separate `.venv-vllm` with vLLM 0.10.2 (cu126) |
| `tests/test_natural.py`, `tests/test_stage1_analysis.py` | Unit + synthetic end-to-end decision scenarios |
| `docs/literature_stage1.md` | Literature table on cost / selection / attribution |

Local tests: **57 passed** (`PYTHONPATH=src .venv\Scripts\python.exe -m pytest -q tests`).

### 2.2 Server and engineering constraints

- Host: `seeklab-01.ece.ncsu.edu` (8× L40S, shared)
- Approved GPUs over the session: initially GPU 3 for pilots; later GPUs 2, 3, 5, 6 for gated Distill work; GPUs 2 and 3 for 7B baselines
- Hugging Face Hub rate-limits the shared IP → `HF_HUB_OFFLINE=1` after first download
- Tokenizers saved by transformers 5 (train env) are unreadable by transformers 4.57 (vLLM env) → eval loads tokenizer from the **base model name**
- Detached pipelines were used so VPN disconnects on the laptop would not kill server jobs

### 2.3 Data builds

| Build | Pool size | Arm A set size | Trace length | Notes |
|---|---|---|---|---|
| First (`data/stage1_v1_4000tok`) | 20,000 | 4,000 | 1,000–4,000 tokens | Used for Pilot 1 |
| Fallback (`data/stage1`) | 11,491 | 3,676 | 1,000–2,500 tokens | Pre-registered fallback after Pilot 1; Arm B subsets overlap more |

Arm A reflection density (fallback build): high ≈ 6.79, low ≈ 2.59, random ≈ 4.59. Leakage check: 0.

---

## 3. Chronology of experiments and results

### 3.1 Pilot 1 — Qwen2.5-1.5B-Instruct, traces ≤ 4,000 tokens

- Run: `armA_random` seed 0 on GPU 3  
- Train: ~500 steps, ~86 min, peak ~19.4 GB  
- Eval: vLLM, greedy, 6144 new tokens  

| Model | GSM8K acc / mean tok / trunc | MATH-500 acc / mean tok / trunc |
|---|---|---|
| Untuned Qwen2.5-1.5B-Instruct | 0.688 / 320 / **0.8%** | 0.504 / 827 / **5.2%** |
| Fine-tuned `armA_random` s0 | 0.420 / 2,878 / **40.6%** | 0.148 / 4,831 / **75.2%** |

**Verdict: FAIL** (truncation ≥ 20%; MATH-500 accuracy below base).

**Diagnosis:** All 203 truncated GSM8K outputs inspected were **degenerate repetition loops**, not long valid reasoning. Non-truncated median ≈ 612 tokens with ~68% accuracy. Raising the generation cap cannot help.

### 3.2 Pilot 2 — same model, pre-registered fallback (traces ≤ 2,500)

| Model | GSM8K | MATH-500 |
|---|---|---|
| Fine-tuned `armA_random` s0 | 0.436 / 2,669 / **36.4%** | 0.150 / 4,491 / **68.6%** |

**Verdict: FAIL.** Same failure mode. Per Stage 1 rule: **no Arm A / Arm B matrix was launched.** The `runs` to-do was cancelled.

### 3.3 Decision after Pilot 2

Options discussed:

1. Start from **DeepSeek-R1-Distill-Qwen-1.5B** and fine-tune lightly (same cost class; risk that base habits dominate / already loops under greedy)
2. Jump to **Qwen2.5-7B-Instruct** (~4–5× cost; not guaranteed to avoid loops)
3. Change decoding (repetition penalty / sampling) — **new pre-registration**
4. Stop Stage 1 and write up template finding + negative result

User asked why Distill would pass if Qwen failed. Answer: Distill is already trained on long R1-style traces, so light SFT might *shift* habits rather than teach a new skill — but Distill is also known to loop under **greedy** decoding (DeepSeek recommends temperature 0.6). So Distill was **not** guaranteed; a cheap untuned baseline check was required first.

### 3.4 Distill-1.5B untuned baseline + gated pipeline

- Config: `configs/stage1_distill.yaml` → `results/stage1_distill`
- GPUs approved: **2, 3, 5, 6**
- Unattended `scripts/gated_distill.sh`: base gate → pilot → (only if PASS) full 29-run matrix
- Later revised so base eval (GPU 2) and pilot train (GPU 3) ran **in parallel** to save wall time; gates still sequential before the matrix

**Untuned DeepSeek-R1-Distill-Qwen-1.5B (greedy):**

| Eval set | Accuracy | Mean tokens | Truncation |
|---|---|---|---|
| GSM8K | 0.328 | 4,078 | **61.6%** |
| MATH-500 | 0.372 | 3,526 | **49.2%** |

**Pipeline status:** `STOPPED_BASE_TRUNCATION_GATE` at 2026-10-03 12:04. Pilot on GPU 3 was killed. Full matrix never started.

### 3.5 Dual 7B untuned baselines (GPUs 2 and 3)

Both models downloaded (~15 GB each) and evaluated offline under the same greedy / 6144 protocol.

| Untuned model | GSM8K acc / tok / trunc | MATH-500 acc / tok / trunc | Gate |
|---|---|---|---|
| Qwen2.5-7B-Instruct | 0.662 / 5,280 / **85.2%** | 0.708 / 1,488 / **15.8%** | **FAIL** (GSM8K) |
| DeepSeek-R1-Distill-Qwen-7B | 0.448 / 3,378 / **49.0%** | 0.372 / 4,461 / **60.4%** | **FAIL** |

Artifacts: `results/stage1_q7`, `results/stage1_d7`.

**Notable asymmetry:** Qwen-7B looks *healthy* on MATH-500 (high accuracy, truncation under 20%) but collapses on GSM8K under the same prompt and decoding. That pattern is unlikely to be “model too small”; it points at **measurement / prompt / stopping behavior**, which motivates the recommendation below.

### 3.6 Scoreboard of Stage 1 measurement attempts

| Attempt | What ran | Outcome |
|---|---|---|
| Qwen 1.5B FT, traces ≤4k | Pilot 1 | FAIL — loops; accuracy collapse |
| Qwen 1.5B FT, traces ≤2.5k | Pilot 2 | FAIL — same |
| Distill 1.5B untuned | Baseline gate | FAIL — 49–62% trunc |
| Qwen 7B untuned | Baseline gate | FAIL — 85% trunc on GSM8K |
| Distill 7B untuned | Baseline gate | FAIL — 49–60% trunc |
| Full Arm A/B matrix | — | **Never launched** (correctly blocked by gates) |

Nothing is currently running on the approved GPUs.

---

## 4. Interpretation (what these failures mean)

### 4.1 What they are *not*

They are **not** yet evidence that training-trace composition does not affect inference cost. The Stage 1 F-test and heuristic-share analysis were never run on a valid measurement substrate.

### 4.2 What they *are*

They are evidence that the **current cost-measurement protocol** is unstable for natural long-CoT SFT validation:

- Plain-text prompt  
- Greedy decoding  
- Hard cap at 6,144 new tokens  
- Outcome = mean generated tokens over the full completion  

Under that protocol:

- Fine-tuning Qwen-1.5B-Instruct on long reflective traces induces **repetition loops** and destroys accuracy.
- Untuned Distill models (1.5B and 7B) already truncate on roughly half or more of questions under greedy.
- Even Qwen-7B-Instruct, which is accurate on MATH-500, hits the cap on most GSM8K items.

So the gate is catching a **broken yardstick**, not answering the scientific question.

### 4.3 Publishability if one model “works” and another fails

Discussed with the user: a clean 7B-only Stage 1 effect *can* be publishable if framed carefully (1.5B failure reported as a scale/stability limit; Gate 0/1 as the small-model positive result). Mixed outcomes weaken the claim and force a narrower scope (“only models stable enough to measure”). Chasing more scales without fixing measurement wastes GPU time.

---

## 5. Recommendation (detailed)

### 5.1 Short version

**Stop chasing model scale.** Do **one cheap diagnostic** on Qwen2.5-7B GSM8K capped outputs. Then either (A) fix prompt/stopping and re-baseline once, or (B) close Stage 1 and write the paper around Gate 0/1 plus this pre-registered negative / blocked measurement.

Do **not** launch the 29-run matrix, Distill sampling, or another model family until that diagnostic is done.

### 5.2 Why this recommendation

1. **Scale already failed the pre-registered rule.** Pilots failed twice at 1.5B; baselines failed at Distill-1.5B, Qwen-7B, and Distill-7B. More GPUs on the same protocol will repeat the same gate failure.
2. **Qwen-7B’s MATH-500 vs GSM8K split is diagnostic gold.** Same model, same decoding, same cap: MATH-500 passes the truncation bar; GSM8K does not. That is the place to learn whether the bug is loops, missing chat template, bad stop tokens, or genuine long CoT.
3. **Changing decoding without inspection is a design change, not a fix.** Temperature 0.6 / multi-sample is scientifically legitimate but requires a **new pre-registration** (noise model, seeds, cost definition). Doing it blindly burns compute and muddies the paper.
4. **The project already has a positive, clean result (Gate 0/1).** A main-track-shaped paper can center template-driven cost, with Stage 1 as an honest blocked natural-data attempt. That is stronger than a paper that ignores a failed yardstick.

### 5.3 Step A — Diagnostic (recommended immediate action)

**Scope:** No training. Read existing Qwen-7B GSM8K generations under `results/stage1_q7/_base/seed0/eval/gsm8k_test/`.

**Questions to answer:**

1. Among truncated (`max_new_tokens` hit) outputs, what fraction are **n-gram / phrase repetition loops** vs **long coherent CoT** that never emits a stop?
2. How often does a `\boxed{...}` appear *before* the cap? If often, full-completion length is a bad cost metric (post-answer ramble).
3. Does the model behave as if it expected a **chat template** (role markers, assistant turn) rather than the plain `"Question: …\nAnswer:\n"` string?
4. Compare a sample of truncated GSM8K vs non-truncated MATH-500 outputs under the same model.

**Resources:** Laptop + SSH read of artifacts; optionally GPU 2 for a tiny re-gen experiment (e.g. 20 questions) if chat-template vs plain prompt needs a controlled A/B. Wall time: ~30 minutes for inspection; &lt;1 hour if a mini A/B is needed.

**Decision table after diagnostic:**

| Finding | Next action |
|---|---|
| Mostly loops | Do not fine-tune under greedy. Either close Stage 1, or (only with new pre-reg) switch to sampling / loop-aware stop and re-baseline. |
| Long valid CoT, no early box | Measurement may be fine; raise cap carefully *or* redefine cost as tokens-until-first-boxed; re-baseline; then one pilot. |
| Boxes early then rambles | Redefine outcome as tokens until first complete `\boxed{...}` (or until stop). Re-baseline; then one pilot. Document as measurement revision. |
| Plain prompt vs chat template fixes GSM8K truncation (&lt;20% both sets) | Adopt chat template for Instruct models; re-baseline Qwen-7B; if pass, **one** pilot (`armA_random` s0) before any matrix. |

### 5.4 Step B — If diagnostic + one fix recovers a valid substrate

Only then:

1. Update README with a short **measurement-revision note** (what changed and why; keep original failed protocols as recorded history).
2. Re-run untuned baseline on **both** eval sets; require truncation &lt; 20%.
3. Run **one** pilot fine-tune; require truncation &lt; 20% and MATH-500 accuracy above that base.
4. Ask the user before any full Arm A/B matrix (29 runs ≈ tens of GPU-hours at 7B).

Prefer **Qwen2.5-7B-Instruct** over Distill-7B for that path if the chat-template / Instruct formatting fix is what works, because Distill’s greedy truncation is high on *both* sets already.

### 5.5 Step C — If diagnostic does not yield a cheap fix: close Stage 1

**Recommended paper framing:**

| Role | Content |
|---|---|
| Main positive result | Gate 0/1: fixed template wording (not reasoning structure) drives 7–17% over-generation on synthetic styles at 0.5B/1.5B |
| Secondary / negative | Natural long-CoT SFT composition cannot be stress-tested for TraCIL under a pre-registered greedy cost protocol at 1.5B–7B; measurement fails via loops / mass truncation |
| Deferred | TraCIL estimator and group-removal validation until a stable cost substrate exists |

This is weaker than a full TraCIL validation paper but is **honest, pre-registered, and defensible**. It is preferable to spending another ~150 GPU-hours on a matrix the gate already forbids.

### 5.6 Explicitly do *not* do next

- Launch the 29-run Arm A/B matrix on any model tried so far  
- Jump to another model family hoping greedy “just works”  
- Switch to temperature 0.6 / multi-sample without (1) inspecting Qwen-7B GSM8K outputs and (2) writing a new pre-registration note  
- Loosen the truncation gate post hoc to force a PASS  
- Treat Distill-1.5B / Distill-7B failures as evidence against the scientific hypothesis rather than against greedy measurement  

### 5.7 Suggested user decision

Approve **Step A only**: inspect Qwen-7B GSM8K capped outputs (and, if needed, a tiny chat-template A/B on GPU 2). After that report, choose Step B (one revised baseline + pilot) or Step C (stop Stage 1 and write).

---

## 6. Current artifact map (server)

| Path | Contents |
|---|---|
| `data/stage1` | Fallback natural pool and Arm A/B trains (≤2,500 tok) |
| `data/stage1_v1_4000tok` | Original ≤4,000 tok build |
| `results/stage1` | Qwen-1.5B Pilot 2 + `_base` |
| `results/stage1_pilot1_4000tok` | Pilot 1 archive |
| `results/stage1_distill` | Distill-1.5B baseline; `PIPELINE_STATUS=STOPPED_BASE_TRUNCATION_GATE` |
| `results/stage1_q7` | Qwen-7B untuned baseline |
| `results/stage1_d7` | Distill-7B untuned baseline |
| `configs/stage1.yaml` | Original Stage 1 config |
| `configs/stage1_distill.yaml` / `stage1_q7.yaml` / `stage1_d7.yaml` | Variant model/result dirs |

Local repo mirrors the scripts, configs, tests, README Stage 1 section, and literature note. Server results are authoritative for numbers in §3.

---

## 7. Bottom line

Stage 1’s **implementation and gates worked as designed**: they correctly blocked a large compute spend when the measurement substrate was invalid. Every greedy baseline or pilot we tried failed the truncation / accuracy gate. The open scientific question (does natural training-trace composition move inference cost?) is **unanswered**, not falsified.

**Recommended next move:** one diagnostic on Qwen-7B GSM8K generations; then either a narrowly scoped measurement fix + single re-pilot, or close Stage 1 and write Gate 0/1 + this blocked natural-data attempt as the paper’s spine.
