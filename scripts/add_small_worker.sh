#!/usr/bin/env bash
# Add one memory-capped small worker on a shared GPU. Does not touch the job list.
# Usage: bash scripts/add_small_worker.sh <gpu> [mem_cap_gb]
set -euo pipefail
cd "$(dirname "$0")/.."
g="$1"; CAP="${2:-12}"
Q=$(cat results/LATEST_QUEUE)
ILA_GPU_MEM_GB=$CAP EXTRA_SET="train.per_device_batch_size=2 train.grad_accum=8 eval.hf_batch_size=16" \
CUDA_VISIBLE_DEVICES=$g setsid nohup bash scripts/queue_worker_small.sh "$Q/jobs_small.txt" "$Q" \
  > "$Q/small_worker_gpu${g}.log" 2>&1 < /dev/null &
echo "small worker on GPU $g (pid $!), cap ${CAP} GB"
