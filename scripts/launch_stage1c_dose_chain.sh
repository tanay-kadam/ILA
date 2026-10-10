#!/usr/bin/env bash
# Follow the existing dose-60 controller, then apply its registered continuation gate.
set -euo pipefail
cd "$(dirname "$0")/.."
controller=results/stage1c_dose_chain_controller
status_file="$controller/STATUS"
gpus="${ILA_STAGE1C_GPUS:-2 3 5 6}"
set_status() { echo "$1 $(date -Is)" | tee "$status_file"; }
run_worker() {
  set_status WAITING_FOR_DOSE60
  while true; do
    state=$(cat results/stage1c_dose60_controller/STAGE1C_DOSE60_STATUS)
    case "$state" in
      COMPLETED*) break ;;
      FAILED*) set_status FAILED_DOSE60; return 20 ;;
    esac
    if ! kill -0 "$(cat logs/stage1c_dose60/controller.pid)" 2>/dev/null; then
      set_status FAILED_DOSE60_CONTROLLER_EXITED; return 21
    fi
    sleep 30
  done
  set_status VERIFYING_DOSE60_ANALYSIS
  PYTHONPATH=src .venv/bin/python scripts/analyze_stage1c_dose60.py --config configs/stage1c_dose60.yaml \
    > "$controller/dose60_analysis.log" 2>&1
  decision=$(.venv/bin/python -c 'import json; print(json.load(open("results/stage1c_dose60/analysis_v2.json"))["decision"])')
  if [[ "$decision" == STOP_QUALITY_FAILURE || "$decision" == STOP_NO_SIGNAL ]]; then
    set_status "COMPLETED_DOSE60_${decision}_ANALYSIS_READY"; return 0
  fi
  test "$decision" == STABLE_SIGNAL_REVIEW
  config=configs/stage1c_dose230.yaml
  results=results/stage1c_dose230
  test ! -e "$results"
  set_status RUNNING_DOSE230_PRECHECK
  for seed in 0 1; do
    if [[ "$seed" == 0 ]]; then source=results/stage1c_q7_lr1e5; else source=results/stage1c_q7_lr1e5_replfix; fi
    PYTHONPATH=src .venv/bin/python scripts/register_adapter_run.py --config "$config" \
      --style armA_random --seed "$seed" --adapter "$source/armA_random/seed$seed/trajectory_adapters/step230"
  done
  queue="$controller/queue"
  mkdir -p "$queue"
  : > "$queue/jobs.txt"
  for seed in 0 1 2; do
    for name in armA_high armA_low armA_random; do
      echo "$config $name $seed" >> "$queue/jobs.txt"
    done
  done
  # Smoke the new endpoint on one step; the full schedule is unchanged.
  set_status RUNNING_DOSE230_SMOKE
  CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/train_sft.py \
    --config "$config" --style armA_high --seed 999 --set results_dir=results/stage1c_dose230_smoke train.stop_after_steps=1
  CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 ILA_VLLM_UTIL=0.85 PYTHONPATH=src \
    .venv-vllm/bin/python scripts/evaluate.py --config "$config" --style armA_high --seed 999 \
    --eval-set gsm8k_test --set results_dir=results/stage1c_dose230_smoke --limit 2
  set_status RUNNING_DOSE230_9_RUNS_7_NEW_TRAINS
  pids=()
  for gpu in $gpus; do
    CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 EVAL_PY=.venv-vllm/bin/python ILA_VLLM_UTIL=0.85 \
      bash scripts/queue_worker.sh "$queue/jobs.txt" "$queue" > "$queue/worker_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if [[ "$failed" != 0 || -n "$(find "$queue/failed" -type f -print -quit)" ]]; then
    set_status FAILED_DOSE230_JOB; return 22
  fi
  set_status RUNNING_DOSE230_ANALYSIS
  PYTHONPATH=src .venv/bin/python scripts/analyze_stage1c_dose60.py --config "$config" \
    > "$results/analysis_stdout.log" 2>&1
  test -f "$results/analysis_v2.json"
  test -f "$results/analysis_v2.md"
  set_status COMPLETED_DOSE230_ANALYSIS_READY
}
if [[ "${1:-}" == --worker ]]; then
  trap 'code=$?; if [[ $code -ne 0 ]]; then set_status "FAILED_EXIT_$code"; fi' EXIT
  run_worker; exit 0
fi
mkdir -p "$controller"
if [[ -f "$controller/controller.pid" ]] && kill -0 "$(cat "$controller/controller.pid")" 2>/dev/null; then
  echo "dose chain already running"; exit 1
fi
nohup bash scripts/launch_stage1c_dose_chain.sh --worker > "$controller/controller.log" 2>&1 < /dev/null &
echo "$!" > "$controller/controller.pid"
echo "dose chain pid=$! status=$status_file"
