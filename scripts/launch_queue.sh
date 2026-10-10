#!/usr/bin/env bash
# Build the job list (longest jobs first) and start one detached worker per GPU.
# Usage: bash scripts/launch_queue.sh "1 2 3 4 5 6 7"
set -euo pipefail
cd "$(dirname "$0")/.."
read -ra GPUS <<< "${1:-1 2 3}"
Q=results/queue_$(date +%Y%m%d_%H%M%S)
mkdir -p "$Q"
: > "$Q/jobs.txt"
for cfg in configs/gate0_1p5b.yaml configs/gate0_para.yaml; do
  for seed in 0 1 2; do
    for style in concise detailed verify backtrack resolve; do
      echo "$cfg $style $seed" >> "$Q/jobs.txt"
    done
  done
done
echo "$(wc -l < "$Q/jobs.txt") jobs -> $Q"
for g in "${GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=$g setsid nohup bash scripts/queue_worker.sh "$Q/jobs.txt" "$Q" \
    > "$Q/worker_gpu${g}.log" 2>&1 < /dev/null &
  echo "worker GPU $g pid $!"
done
echo "$Q" > results/LATEST_QUEUE
