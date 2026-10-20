#!/usr/bin/env bash
# Evaluate one finished run with the vLLM env. Usage: CUDA_VISIBLE_DEVICES=3 bash scripts/eval_run.sh <set> <seed>
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 ILA_VLLM_UTIL="${ILA_VLLM_UTIL:-0.45}"
for es in gsm8k_test math500; do
  .venv-vllm/bin/python scripts/evaluate.py --config configs/stage1.yaml --style "$1" --seed "$2" --eval-set "$es" \
    > "results/stage1/logs/eval_${1}_seed${2}_${es}.log" 2>&1
done
echo EVALS_DONE > "results/stage1/eval_${1}_seed${2}.flag"
