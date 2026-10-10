#!/usr/bin/env bash
# Optimized four-GPU Stage 1c matrix. Runs analysis but never launches Stage 2.
set -euo pipefail

cd "$(dirname "$0")/.."

config="configs/stage1c_matrix.yaml"
results="results/stage1c_matrix"
smoke="results/stage1c_matrix_smoke"
controller="results/stage1c_matrix_controller"
log_dir="logs/stage1c_matrix"
log="$log_dir/controller.log"
pid_file="$log_dir/controller.pid"
status_file="$controller/STAGE1C_MATRIX_STATUS"
gpus="${ILA_STAGE1C_GPUS:-2 3 5 6}"

set_status() {
  mkdir -p "$controller"
  echo "$1 $(date -Is)" | tee "$status_file"
}

write_jobs() {
  local jobs="$1"
  : > "$jobs"
  echo "$config armA_random 0" >> "$jobs"
  for name in armA_high armA_low; do echo "$config $name 0" >> "$jobs"; done
  for seed in 1 2; do
    for name in armA_random armA_high armA_low; do echo "$config $name $seed" >> "$jobs"; done
  done
  for j in $(seq -w 0 15); do echo "$config b$j 0" >> "$jobs"; done
  for seed in 1 2; do
    for name in b00 b01; do echo "$config $name $seed" >> "$jobs"; done
  done
  test "$(wc -l < "$jobs")" -eq 29
}

run_worker() {
  echo "STARTED $(date -Is) host=$(hostname) physical_gpus=$gpus"
  set_status "RUNNING_PRECHECK"
  test -f docs/stage1c_matrix_execution.md
  test -f results/stage1_q7_chat/_base/seed0/eval/gsm8k_test/outputs.jsonl
  test -f results/stage1_q7_chat/_base/seed0/eval/math500/outputs.jsonl
  seed0_adapter="results/stage1c_q7_lr1e5/armA_random/seed0/trajectory_adapters/step015"
  seed1_adapter="results/stage1c_q7_lr1e5_replfix/armA_random/seed1/trajectory_adapters/step015"
  test -f "$seed0_adapter/adapter_config.json"
  test -f "$seed1_adapter/adapter_config.json"
  if [[ -e "$results" || -e "$smoke" ]]; then
    echo "Matrix or smoke output already exists; refusing to overwrite" >&2
    return 10
  fi
  for name in armA_high armA_low armA_random $(seq -f 'b%02g' 0 15); do
    test -f "data/stage1/train_${name}.jsonl"
  done

  set_status "RUNNING_SMOKE"
  CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 PYTHONPATH=src \
    .venv/bin/python scripts/train_sft.py \
      --config "$config" --style armA_random --seed 999 \
      --set results_dir="$smoke" train.stop_after_steps=1
  test -f "$smoke/armA_random/seed999/adapter/adapter_config.json"
  test ! -d "$smoke/armA_random/seed999/model"
  CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 ILA_VLLM_UTIL=0.85 PYTHONPATH=src \
    .venv-vllm/bin/python scripts/evaluate.py \
      --config "$config" --style armA_random --seed 999 --eval-set gsm8k_test \
      --set results_dir="$smoke" --limit 2

  set_status "SMOKE_PASSED_PREPARING_MATRIX"
  mkdir -p "$results/_base/seed0"
  cp -a results/stage1_q7_chat/_base/seed0/eval "$results/_base/seed0/eval"
  PYTHONPATH=src .venv/bin/python scripts/register_adapter_run.py \
    --config "$config" --style armA_random --seed 0 --adapter "$seed0_adapter"
  PYTHONPATH=src .venv/bin/python scripts/register_adapter_run.py \
    --config "$config" --style armA_random --seed 1 --adapter "$seed1_adapter"
  queue="$controller/queue"
  mkdir -p "$queue"
  write_jobs "$queue/jobs.txt"
  set_status "RUNNING_MATRIX_29_RUNS_27_NEW_TRAINS"

  worker_pids=()
  for gpu in $gpus; do
    CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 EVAL_PY=.venv-vllm/bin/python \
      ILA_VLLM_UTIL=0.85 bash scripts/queue_worker.sh "$queue/jobs.txt" "$queue" \
      > "$queue/worker_gpu${gpu}.log" 2>&1 &
    worker_pids+=("$!")
    echo "WORKER gpu=$gpu pid=$!"
  done
  worker_failure=0
  for worker_pid in "${worker_pids[@]}"; do
    wait "$worker_pid" || worker_failure=1
  done
  if [[ "$worker_failure" -ne 0 || -n "$(find "$queue/failed" -type f -print -quit)" ]]; then
    set_status "FAILED_MATRIX_WORKER_OR_JOB"
    return 20
  fi

  completed=$(find "$results" -path '*/eval/math500/summary.json' ! -path '*/_base/*' | wc -l)
  if [[ "$completed" -ne 29 ]]; then
    echo "expected 29 completed math500 summaries, found $completed" >&2
    set_status "FAILED_INCOMPLETE_ARTIFACTS_$completed"
    return 21
  fi
  set_status "RUNNING_ANALYSIS"
  PYTHONPATH=src .venv/bin/python scripts/analyze_stage1.py --config "$config" \
    > "$results/analysis_stdout.log" 2>&1
  test -f "$results/analysis.json"
  set_status "COMPLETED_MATRIX_AND_ANALYSIS_PENDING_REVIEW"
}

if [[ "${1:-}" == "--worker" ]]; then
  trap 'code=$?; if [[ $code -ne 0 ]]; then set_status "FAILED_EXIT_$code"; fi' EXIT
  run_worker
  exit 0
fi

mkdir -p "$log_dir" "$controller"
if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  echo "Stage 1c matrix is already running as PID $(cat "$pid_file")"
  exit 1
fi
nohup "$0" --worker >"$log" 2>&1 < /dev/null &
echo "$!" >"$pid_file"
echo "Stage 1c matrix controller pid=$! gpus=$gpus log=$log status=$status_file"
