#!/usr/bin/env bash
# Stop the queue worker bound to one GPU, release its in-flight job, and (optionally)
# start a replacement worker on another GPU.
# Usage: bash scripts/drop_gpu.sh <gpu_to_drop> [gpu_to_add]
set -uo pipefail
cd "$(dirname "$0")/.."
DROP="$1"; ADD="${2:-}"
Q=$(cat results/LATEST_QUEUE)
echo "queue: $Q"

WPID=""
for pid in $(pgrep -f 'queue_worker'); do
  if tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$DROP"; then WPID=$pid; fi
done
if [[ -z "$WPID" ]]; then echo "no worker found on GPU $DROP"; else
  LOGF=$(ls -t "$Q"/*worker_gpu${DROP}*.log | head -1)
  JOB=$(grep ' start ' "$LOGF" | tail -1 | sed 's/.* start //')
  echo "worker pid $WPID, in-flight job: ${JOB:-none}"
  kill -STOP "$WPID"
  pkill -TERM -P "$WPID"
  kill -TERM "$WPID"; kill -CONT "$WPID" 2>/dev/null
  sleep 3
  pkill -9 -P "$WPID" 2>/dev/null
  if [[ -n "${JOB:-}" ]] && ! grep -q "done  $JOB" "$LOGF"; then
    cfg=$(echo "$JOB" | sed 's/_\(concise\|detailed\|verify\|backtrack\|resolve\)_seed.*//')
    style=$(echo "$JOB" | sed 's/.*_\(concise\|detailed\|verify\|backtrack\|resolve\)_seed.*/\1/')
    seed=$(echo "$JOB" | sed 's/.*_seed//')
    res=$(.venv/bin/python -c "import sys; sys.path.insert(0,'src'); from ila.config import load_config; print(load_config('configs/${cfg}.yaml')['results_dir'])")
    rm -rf "$res/$style/seed$seed" "$Q/claims/$JOB"
    echo "released $JOB (removed partial $res/$style/seed$seed)"
  fi
fi
sleep 2
echo "processes still bound to GPU $DROP:"
nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader
if [[ -n "$ADD" ]]; then
  CUDA_VISIBLE_DEVICES=$ADD setsid nohup bash scripts/queue_worker.sh "$Q/jobs.txt" "$Q" \
    > "$Q/worker_gpu${ADD}_b.log" 2>&1 < /dev/null &
  echo "started replacement worker on GPU $ADD"
fi
