#!/usr/bin/env bash
# Independent detached branches; physical GPUs 5 and 6 are never permitted.
set -euo pipefail
cd "$(dirname "$0")/.."
controller=results/tmlr_strengthening_controller
mkdir -p "$controller"
status() { echo "$2 $(date -Is)" > "$controller/${1}_STATUS"; }
run_queue() {
  local family="$1"
  local queue="$controller/${family}_queue"
  mkdir -p "$queue"
  PYTHONPATH=src .venv/bin/python scripts/write_tmlr_jobs.py --family "$family" --output "$queue/jobs.json"
  pids=()
  for gpu in 0 1 2 3 4 7; do
    PYTHONPATH=src .venv/bin/python scripts/tmlr_gpu_worker.py --jobs "$queue/jobs.json" --queue "$queue" --gpu "$gpu" > "$queue/worker_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  failure=0
  for pid in "${pids[@]}"; do wait "$pid" || failure=1; done
  test "$failure" -eq 0
  test -z "$(find "$queue/failed" -type f -print -quit)"
  PYTHONPATH=src .venv/bin/python scripts/analyze_tmlr_external.py --family "$family" > "$controller/${family}_analysis.log" 2>&1
  status "$family" COMPLETED_ANALYSIS_READY
}
branch() {
  local family="$1"
  trap 'code=$?; if [[ $code -ne 0 ]]; then status "$family" "FAILED_EXIT_$code"; fi' EXIT
  status "$family" RUNNING_PRECHECK
  test -f data/tmlr_external/manifest.json
  if [[ "$family" == mistral ]]; then
    status "$family" PINNING_MODEL_AND_SERIALIZATION_PRECHECK
    PYTHONPATH=src .venv/bin/python scripts/prepare_tmlr_mistral.py
    status "$family" RUNNING_TRAIN_AND_EVAL_SMOKE
    # Wait for capacity rather than sharing a busy GPU with another user's job.
    PYTHONPATH=src .venv/bin/python scripts/tmlr_smoke.py --family mistral
  else
    status "$family" RUNNING_EVAL_SMOKE
    PYTHONPATH=src .venv/bin/python scripts/tmlr_smoke.py --family qwen
  fi
  status "$family" SMOKE_PASSED_RUNNING_FIXED_QUEUE
  run_queue "$family"
}
if [[ "${1:-}" == --branch ]]; then branch "$2"; exit 0; fi
for family in qwen mistral; do
  pid_file="$controller/${family}.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$family branch already running; preserved"; continue
  fi
  if [[ -d "$controller/${family}_queue" ]]; then
    echo "$family queue already exists; inspect before restarting" >&2; exit 10
  fi
  nohup bash scripts/launch_tmlr_strengthening.sh --branch "$family" > "$controller/${family}.log" 2>&1 < /dev/null &
  echo "$!" > "$pid_file"
  echo "$family branch pid=$! gpu_allowlist=0,1,2,3,4,7"
done
