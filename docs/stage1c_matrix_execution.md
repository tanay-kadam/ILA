# Stage 1c natural-composition matrix execution addendum

**Registered:** 2026-10-06, after the corrected seed-1 replication passed and before
any Stage 1c matrix arm other than `armA_random` was trained.

## Authorization and purpose

The canonical-chat Qwen2.5-7B substrate passed the stability gate under two valid
training seeds at learning rate `1e-5`, optimizer step 15 of a cosine schedule planned
for 230 steps. This addendum authorizes the already designed 29-run Arm A/B natural
composition matrix using that frozen recipe. It does not alter the original Arm A/B
subset assignments, evaluation questions, decoding, outcomes, or decision rules.

## Fixed training implementation

- Model: `Qwen/Qwen2.5-7B-Instruct`.
- Prompt: canonical chat, completion-only assistant loss.
- LoRA: rank 64, alpha 128, dropout 0.05, all seven configured projection families.
- Learning rate: `1e-5`; cosine scheduler planned for 230 optimizer steps.
- Actual optimization stops immediately after step 15 through a Trainer callback.
  Keeping `max_steps=230` preserves the validated learning-rate trajectory; changing
  `max_steps` to 15 is prohibited.
- Batch size, accumulation, max length, BF16, seed handling, and all other training
  settings remain unchanged.
- Save adapter weights only. Evaluation uses vLLM dynamic LoRA, matching the validated
  Stage 1c checkpoints. No merged model is produced per run.

The already validated `armA_random` seed-0 and corrected seed-1 step-15 adapters are
reused and evaluated in the matrix output tree. Every other run is newly trained.

## Matrix and evaluation

- Arm A: high/low/random × seeds 0, 1, 2 = 9 runs.
- Arm B: b00–b15 at seed 0 = 16 runs; b00 and b01 at seeds 1 and 2 = 4 runs.
- Total: 29 evaluated runs, 27 new training runs.
- Evaluation: the fixed GSM8K-500 and MATH-500 sets, greedy decoding, 6,144-token cap.
- The canonical-chat full baseline is reused from the recorded Stage 1b baseline.

## Analysis and stopping

Run the existing preregistered `scripts/analyze_stage1.py` procedure unchanged:
pooled within-subset seed noise, one-sided F test, minimum 5% relative cost range,
leave-one-out ridge heuristic share, and the fixed STOP/GO/AMBIGUOUS thresholds.

All failed runs and partial artifacts are retained. The controller may analyze only if
all 29 runs complete successfully. No Stage 2 work starts automatically regardless of
the matrix verdict.

