#!/usr/bin/env bash
# One final recipe: engineering gate -> conditional three-seed comparison -> analysis.
set -euo pipefail
cd "$(dirname "$0")/.."
config=configs/stage1c_final_stability.yaml
results=results/stage1c_final_stability
smoke=results/stage1c_final_stability_smoke
controller=results/stage1c_final_stability_controller
gpu="${ILA_STAGE1C_GPU:-3}"
status_file="$controller/STATUS"
set_status() { echo "$1 $(date -Is)" | tee "$status_file"; }
train() { CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/train_sft.py --config "$config" --style "$1" --seed "$2"; }
evaluate() { CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 ILA_VLLM_UTIL=0.85 PYTHONPATH=src .venv-vllm/bin/python scripts/evaluate.py --config "$config" --style "$1" --seed "$2" --eval-set "$3"; }
worker() {
  set_status RUNNING_PRECHECK
  test ! -e "$results"
  test ! -e "$smoke"
  test -f docs/stage1c_final_stability.md
  test -f data/stage1/train_armA_high.jsonl
  test -f data/stage1/train_armA_low.jsonl
  test -f data/stage1/train_armA_random.jsonl
  echo "host=$(hostname) physical_gpu=$gpu config=$config"
  PYTHONPATH=src .venv/bin/python scripts/materialize_stage1c_splits.py --config "$config" --source-results results/stage1_q7_chat
  mkdir -p "$results/_base/seed0/eval" "$results/logs"
  cp -a results/stage1c_matrix/_base/seed0/eval/gsm8k_test "$results/_base/seed0/eval/"
  cp -a results/stage1c_matrix/_base/seed0/eval/math500 "$results/_base/seed0/eval/"
  set_status RUNNING_SMOKE
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/train_sft.py \
    --config "$config" --style armA_random --seed 999 --set results_dir="$smoke" train.stop_after_steps=1
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 ILA_VLLM_UTIL=0.85 PYTHONPATH=src .venv-vllm/bin/python scripts/evaluate.py \
    --config "$config" --style armA_random --seed 999 --eval-set gsm8k_dev100 --limit 2 --set results_dir="$smoke"
  set_status SMOKE_PASSED_RUNNING_ENGINEERING_PILOTS
  for style in armA_random armA_high armA_low; do
    train "$style" 0 > "$results/logs/train_${style}_seed0.log" 2>&1
    for es in gsm8k_dev100 math500_dev100; do
      evaluate "$style" 0 "$es" > "$results/logs/eval_${style}_seed0_${es}.log" 2>&1
    done
    if PYTHONPATH=src .venv/bin/python scripts/gate_final_stability.py --phase engineering --style "$style" --seed 0; then
      echo "ENGINEERING_PASS $style"
    else
      code=$?
      if [[ "$code" -ne 2 ]]; then return "$code"; fi
      PYTHONPATH=src .venv/bin/python scripts/gate_final_stability.py --phase report
      set_status COMPLETED_STOP_QUALITY_FAILURE_ANALYSIS_READY
      return 0
    fi
  done
  set_status ENGINEERING_PASSED_RUNNING_THREE_SEED_COMPARISON
  queue="$controller/queue"
  mkdir -p "$queue"
  : > "$queue/jobs.txt"
  for seed in 0 1 2; do
    for style in armA_high armA_low armA_random; do
      echo "$config $style $seed" >> "$queue/jobs.txt"
    done
  done
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 EVAL_PY=.venv-vllm/bin/python ILA_VLLM_UTIL=0.85 \
    bash scripts/queue_worker.sh "$queue/jobs.txt" "$queue" > "$queue/worker_gpu${gpu}.log" 2>&1
  if [[ -n "$(find "$queue/failed" -type f -print -quit)" ]]; then
    set_status FAILED_COMPARISON_JOB; return 20
  fi
  set_status RUNNING_CONFIRMATION_AND_ANALYSIS
  for seed in 0 1 2; do
    for style in armA_high armA_low armA_random; do
      if PYTHONPATH=src .venv/bin/python scripts/gate_final_stability.py --phase confirmation --style "$style" --seed "$seed"; then
        echo "CONFIRMATION_PASS $style seed=$seed"
      else
        code=$?; if [[ "$code" -ne 2 ]]; then return "$code"; fi
      fi
    done
  done
  PYTHONPATH=src .venv/bin/python scripts/analyze_stage1c_dose60.py --config "$config" > "$results/analysis_stdout.log" 2>&1
  PYTHONPATH=src .venv/bin/python scripts/gate_final_stability.py --phase report
  test -f "$results/final_report.json"
  set_status COMPLETED_FINAL_STABILITY_ANALYSIS_READY
}
if [[ "${1:-}" == --worker ]]; then
  trap 'code=$?; if [[ $code -ne 0 ]]; then set_status "FAILED_EXIT_$code"; fi' EXIT
  worker; exit 0
fi
mkdir -p "$controller"
if [[ -f "$controller/controller.pid" ]] && kill -0 "$(cat "$controller/controller.pid")" 2>/dev/null; then
  echo "final stability controller already running"; exit 1
fi
nohup bash scripts/launch_stage1c_final_stability.sh --worker > "$controller/controller.log" 2>&1 < /dev/null &
echo "$!" > "$controller/controller.pid"
echo "final stability pid=$! gpu=$gpu status=$status_file"
