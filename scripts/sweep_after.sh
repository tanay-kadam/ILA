#!/usr/bin/env bash
# For each given GPU: wait for its current queue worker to exit, then run a fresh worker
# on that GPU so jobs released after the workers passed them still get executed.
# Claims are atomic, so only unclaimed jobs run. Usage: bash scripts/sweep_after.sh 1 2 3
set -uo pipefail
cd "$(dirname "$0")/.."
Q=$(cat results/LATEST_QUEUE)
for g in "$@"; do
  wpid=""
  for pid in $(pgrep -f 'queue_worker.sh'); do
    if tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$g"; then wpid=$pid; fi
  done
  [[ -z "$wpid" ]] && { echo "GPU $g: no active worker, skipping"; continue; }
  setsid nohup bash -c "while kill -0 $wpid 2>/dev/null; do sleep 30; done; CUDA_VISIBLE_DEVICES=$g bash scripts/queue_worker.sh '$Q/jobs.txt' '$Q'" \
    > "$Q/sweep_gpu${g}.log" 2>&1 < /dev/null &
  echo "GPU $g: sweeper armed behind worker pid $wpid"
done
