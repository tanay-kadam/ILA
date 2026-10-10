#!/usr/bin/env bash
# Run the 0.5B paraphrase extra-seed jobs (seeds 3-5) with small, memory-capped workers on
# GPUs that other jobs are using. Effective batch stays 16 (2 x 8 accumulation), but batch
# composition per step differs from the seed 0-2 runs; the resolved overrides are stored in
# each run_record.json so these runs can be separated in analysis.
# Usage: bash scripts/launch_small.sh "3 5 6 7" [mem_cap_gb]
set -euo pipefail
cd "$(dirname "$0")/.."
read -ra GPUS <<< "$1"
CAP="${2:-12}"
Q=$(cat results/LATEST_QUEUE)
grep '^configs/gate0_para.yaml ' "$Q/jobs.txt" > "$Q/jobs_small.txt"
echo "$(wc -l < "$Q/jobs_small.txt") small jobs, cap ${CAP} GB"
for g in "${GPUS[@]}"; do
  ILA_GPU_MEM_GB=$CAP EXTRA_SET="train.per_device_batch_size=2 train.grad_accum=8 eval.hf_batch_size=16" \
  CUDA_VISIBLE_DEVICES=$g setsid nohup bash scripts/queue_worker_small.sh "$Q/jobs_small.txt" "$Q" \
    > "$Q/small_worker_gpu${g}.log" 2>&1 < /dev/null &
  echo "small worker on GPU $g (pid $!)"
done
