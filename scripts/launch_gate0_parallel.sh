#!/usr/bin/env bash
# Launch one Gate 0 seed shard per GPU in the background (survives logout).
# Usage: bash scripts/launch_gate0_parallel.sh "1 2 3" "0 1 2"
#        (GPU ids, then seeds; paired in order)
set -euo pipefail
cd "$(dirname "$0")/.."
read -ra GPUS <<< "${1:-1 2 3}"
read -ra SEEDS <<< "${2:-0 1 2}"
mkdir -p results/gate0/logs
export HF_HUB_DISABLE_PROGRESS_BARS=1 PATH="$PWD/.venv/bin:$PATH"
for i in "${!SEEDS[@]}"; do
  g=${GPUS[$i]}; s=${SEEDS[$i]}
  CUDA_VISIBLE_DEVICES=$g SEEDS=$s SKIP_ANALYSIS=1 \
    setsid nohup bash scripts/run_gate0.sh configs/gate0.yaml \
    > "results/gate0/logs/shard_seed${s}_gpu${g}.log" 2>&1 < /dev/null &
  echo "seed $s -> GPU $g (pid $!)"
done
