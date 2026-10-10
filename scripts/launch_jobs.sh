#!/usr/bin/env bash
# Create the follow-up queue (30 jobs, longest first) and start one worker per given GPU.
#   1) paraphrased styles on Qwen2.5-1.5B, seeds 0-2        (15 jobs)
#   2) paraphrased styles on Qwen2.5-0.5B, extra seeds 3-5  (15 jobs)
# Workers claim jobs atomically, so GPUs can be added later with scripts/add_worker.sh.
# Usage: bash scripts/launch_jobs.sh "5 6"      (use "" to create the queue with no workers)
set -euo pipefail
cd "$(dirname "$0")/.."
read -ra GPUS <<< "${1:-}"
Q=results/queue_$(date +%Y%m%d_%H%M%S)
mkdir -p "$Q"
: > "$Q/jobs.txt"
for seed in 0 1 2; do
  for style in concise detailed verify backtrack resolve; do echo "configs/gate0_para_1p5b.yaml $style $seed" >> "$Q/jobs.txt"; done
done
for seed in 3 4 5; do
  for style in concise detailed verify backtrack resolve; do echo "configs/gate0_para.yaml $style $seed" >> "$Q/jobs.txt"; done
done
echo "$Q" > results/LATEST_QUEUE
echo "$(wc -l < "$Q/jobs.txt") jobs -> $Q"
for g in "${GPUS[@]:-}"; do
  [[ -z "$g" ]] && continue
  CUDA_VISIBLE_DEVICES=$g setsid nohup bash scripts/queue_worker.sh "$Q/jobs.txt" "$Q" \
    > "$Q/worker_gpu${g}.log" 2>&1 < /dev/null &
  echo "worker on GPU $g (pid $!)"
done
