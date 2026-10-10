#!/usr/bin/env bash
# Launch the preregistered Stage 1b canonical-chat baseline so it survives SSH/VPN drops.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs/stage1b_baseline

launch_one() {
  local gpu="$1"
  local eval_set="$2"
  local log="logs/stage1b_baseline/${eval_set}.log"
  local pid_file="logs/stage1b_baseline/${eval_set}.pid"

  CUDA_VISIBLE_DEVICES="$gpu" \
  HF_HUB_OFFLINE=1 \
  ILA_VLLM_UTIL=0.85 \
  PYTHONPATH=src \
    nohup .venv-vllm/bin/python scripts/evaluate.py \
      --config configs/stage1_q7.yaml \
      --style _base \
      --seed 0 \
      --eval-set "$eval_set" \
      --base \
      --chat-template \
      --force \
      --set results_dir=results/stage1_q7_chat \
      >"$log" 2>&1 < /dev/null &
  echo "$!" >"$pid_file"
  echo "$eval_set pid=$! gpu=$gpu log=$log"
}

launch_one 2 gsm8k_test
launch_one 3 math500
