#!/usr/bin/env bash
# Short end-to-end check on one GPU before launching the full matrix.
# Usage: CUDA_VISIBLE_DEVICES=1 bash scripts/gpu_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=results/gpu_smoke
rm -rf "$OUT"
export HF_HUB_DISABLE_PROGRESS_BARS=1

SECONDS=0
$PY scripts/train_sft.py --config configs/gate0.yaml --style verify --seed 0 \
  --set train.max_steps=20 results_dir=$OUT
echo "TRAIN_20_STEPS_SECONDS=$SECONDS"

SECONDS=0
$PY scripts/evaluate.py --config configs/gate0.yaml --style verify --seed 0 \
  --eval-set gsm8k_test --limit 128 --set results_dir=$OUT
echo "EVAL_128_SECONDS=$SECONDS"

$PY - <<'EOF'
import json
r = json.load(open("results/gpu_smoke/verify/seed0/run_record.json"))
print("RECORD", r["status"], r["device"], r["versions"], "n_train", r["n_train"], "truncated", r["n_truncated_examples"])
for o in [json.loads(l) for l in open("results/gpu_smoke/verify/seed0/eval/gsm8k_test/outputs.jsonl")][:2]:
    print("----", o["n_gen_tokens"], o["truncated"], o["pred"], o["gold"])
    print(o["output"][:600])
EOF
echo SMOKE_DONE
