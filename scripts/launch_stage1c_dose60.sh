#!/usr/bin/env bash
# Budget-capped dose-60 follow-up. It cannot launch the step-230 phase.
set -euo pipefail

cd "$(dirname "$0")/.."

config="configs/stage1c_dose60.yaml"
results="results/stage1c_dose60"
smoke="results/stage1c_dose60_smoke"
controller="results/stage1c_dose60_controller"
log_dir="logs/stage1c_dose60"
log="$log_dir/controller.log"
pid_file="$log_dir/controller.pid"
status_file="$controller/STAGE1C_DOSE60_STATUS"
gpus="${ILA_STAGE1C_GPUS:-2 3 5 6}"

set_status() { mkdir -p "$controller"; echo "$1 $(date -Is)" | tee "$status_file"; }

run_worker() {
  echo "STARTED $(date -Is) host=$(hostname) physical_gpus=$gpus"
  set_status "RUNNING_PRECHECK"
  test -f docs/stage1c_dose_response.md
  test -f configs/stage1c_dose60.yaml
  test -f results/stage1c_q7_lr1e5/armA_random/seed0/trajectory_adapters/step060/adapter_config.json
  test -f results/stage1c_q7_lr1e5_replfix/armA_random/seed1/trajectory_adapters/step060/adapter_config.json
  test -f results/stage1c_matrix/_base/seed0/eval/gsm8k_test/outputs.jsonl
  test -f results/stage1c_matrix/_base/seed0/eval/math500/outputs.jsonl
  if [[ -e "$results" || -e "$smoke" ]]; then
    echo "dose-60 result or smoke directory exists; refusing to overwrite" >&2
    return 10
  fi

  set_status "RUNNING_SMOKE"
  CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/train_sft.py \
    --config "$config" --style armA_high --seed 999 \
    --set results_dir="$smoke" train.stop_after_steps=1
  test -f "$smoke/armA_high/seed999/adapter/adapter_config.json"
  CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 ILA_VLLM_UTIL=0.85 PYTHONPATH=src \
    .venv-vllm/bin/python scripts/evaluate.py --config "$config" --style armA_high \
    --seed 999 --eval-set gsm8k_test --set results_dir="$smoke" --limit 2

  set_status "SMOKE_PASSED_PREPARING_DOSE60"
  mkdir -p "$results/_base/seed0"
  cp -a results/stage1c_matrix/_base/seed0/eval "$results/_base/seed0/eval"
  PYTHONPATH=src .venv/bin/python scripts/register_adapter_run.py --config "$config" \
    --style armA_random --seed 0 \
    --adapter results/stage1c_q7_lr1e5/armA_random/seed0/trajectory_adapters/step060
  PYTHONPATH=src .venv/bin/python scripts/register_adapter_run.py --config "$config" \
    --style armA_random --seed 1 \
    --adapter results/stage1c_q7_lr1e5_replfix/armA_random/seed1/trajectory_adapters/step060

  for seed in 0 1; do
    for es in gsm8k_test math500; do
      CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 ILA_VLLM_UTIL=0.85 PYTHONPATH=src \
        .venv-vllm/bin/python scripts/evaluate.py --config "$config" \
        --style armA_random --seed "$seed" --eval-set "$es"
    done
  done

  queue="$controller/queue"
  mkdir -p "$queue"
  : > "$queue/jobs.txt"
  echo "$config armA_random 2" >> "$queue/jobs.txt"
  for seed in 0 1 2; do
    echo "$config armA_high $seed" >> "$queue/jobs.txt"
    echo "$config armA_low $seed" >> "$queue/jobs.txt"
  done
  test "$(wc -l < "$queue/jobs.txt")" -eq 7
  set_status "RUNNING_DOSE60_9_RUNS_7_NEW_TRAINS"

  worker_pids=()
  for gpu in $gpus; do
    CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 EVAL_PY=.venv-vllm/bin/python \
      ILA_VLLM_UTIL=0.85 bash scripts/queue_worker.sh "$queue/jobs.txt" "$queue" \
      > "$queue/worker_gpu${gpu}.log" 2>&1 &
    worker_pids+=("$!")
  done
  failed=0
  for pid in "${worker_pids[@]}"; do wait "$pid" || failed=1; done
  if [[ "$failed" -ne 0 || -n "$(find "$queue/failed" -type f -print -quit)" ]]; then
    set_status "FAILED_DOSE60_WORKER_OR_JOB"; return 20
  fi
  completed=$(find "$results" -path '*/eval/math500/summary.json' ! -path '*/_base/*' | wc -l)
  test "$completed" -eq 9
  set_status "RUNNING_REGISTERED_ANALYSIS"
  PYTHONPATH=src .venv/bin/python scripts/analyze_stage1c_dose60.py --config "$config" \
    > "$results/analysis_stdout.log" 2>&1
  test -f "$results/analysis_v2.json"
  set_status "COMPLETED_DOSE60_PENDING_REVIEW_NO_STEP230_AUTO_LAUNCH"
}

if [[ "${1:-}" == "--worker" ]]; then
  trap 'code=$?; if [[ $code -ne 0 ]]; then set_status "FAILED_EXIT_$code"; fi' EXIT
  run_worker
  exit 0
fi

mkdir -p "$log_dir" "$controller"
if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  echo "dose-60 controller already running as PID $(cat "$pid_file")"; exit 1
fi
nohup "$0" --worker >"$log" 2>&1 < /dev/null &
echo "$!" >"$pid_file"
echo "dose-60 controller pid=$! gpus=$gpus log=$log status=$status_file"
