#!/usr/bin/env bash
# Stop the queue worker on each given GPU (releasing its in-flight job). Usage: bash scripts/drop_many.sh 3 4 5
cd "$(dirname "$0")/.."
for g in "$@"; do
  echo "=== GPU $g ==="
  bash scripts/drop_gpu.sh "$g" 2>&1 | grep -E 'worker pid|no worker|released'
done
echo "=== remaining queue workers ==="
for pid in $(pgrep -f 'queue_worker'); do
  echo "pid $pid $(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -E '^(CUDA_VISIBLE_DEVICES|ILA_GPU_MEM_GB)=' | tr '\n' ' ')"
done
