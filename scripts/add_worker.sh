#!/usr/bin/env bash
# Start an extra worker on a GPU for the latest queue. Usage: bash scripts/add_worker.sh <gpu>
set -euo pipefail
cd "$(dirname "$0")/.."
g="$1"
Q=$(cat results/LATEST_QUEUE)
CUDA_VISIBLE_DEVICES=$g setsid nohup bash scripts/queue_worker.sh "$Q/jobs.txt" "$Q" \
  > "$Q/worker_gpu${g}_$(date +%H%M%S).log" 2>&1 < /dev/null &
echo "worker on GPU $g (pid $!) for $Q"
