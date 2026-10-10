#!/usr/bin/env bash
# Train and evaluate the single preregistered Stage 1b pilot; never launches the matrix.
set -euo pipefail

cd "$(dirname "$0")/.."
log_dir="logs/stage1b_pilot"
log="$log_dir/pipeline.log"
pid_file="$log_dir/pipeline.pid"
status_file="results/stage1_q7_chat/STAGE1B_PILOT_STATUS"

run_worker() {
  echo "STARTED $(date -Is) host=$(hostname) physical_gpu=2"
  echo "TRAIN config=configs/stage1b_q7.yaml style=armA_random seed=0"
  CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 PYTHONPATH=src \
    .venv/bin/python scripts/train_sft.py \
      --config configs/stage1b_q7.yaml \
      --style armA_random \
      --seed 0

  echo "EVAL gsm8k_test $(date -Is)"
  CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 ILA_VLLM_UTIL=0.85 PYTHONPATH=src \
    .venv-vllm/bin/python scripts/evaluate.py \
      --config configs/stage1b_q7.yaml \
      --style armA_random \
      --seed 0 \
      --eval-set gsm8k_test

  echo "EVAL math500 $(date -Is)"
  CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 ILA_VLLM_UTIL=0.85 PYTHONPATH=src \
    .venv-vllm/bin/python scripts/evaluate.py \
      --config configs/stage1b_q7.yaml \
      --style armA_random \
      --seed 0 \
      --eval-set math500

  echo "GATE $(date -Is)"
  PYTHONPATH=src .venv/bin/python scripts/analyze_stage1.py \
    --config configs/stage1b_q7.yaml \
    --pilot | tee results/stage1_q7_chat/stage1b_pilot_gate.json
  echo "COMPLETED $(date -Is)" | tee "$status_file"
}

if [[ "${1:-}" == "--worker" ]]; then
  trap 'code=$?; if [[ $code -ne 0 ]]; then echo "FAILED exit=$code $(date -Is)" | tee "$status_file"; fi' EXIT
  run_worker
  exit 0
fi

mkdir -p "$log_dir" "$(dirname "$status_file")"
if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  echo "Pilot is already running as PID $(cat "$pid_file")"
  exit 1
fi
if [[ -e results/stage1_q7_chat/armA_random/seed0/run_record.json ]]; then
  echo "Pilot output already exists; inspect it before relaunching" >&2
  exit 1
fi
test -f results/stage1_q7_chat/_base/seed0/eval/gsm8k_test/summary.json
test -f results/stage1_q7_chat/_base/seed0/eval/math500/summary.json

nohup "$0" --worker >"$log" 2>&1 < /dev/null &
echo "$!" >"$pid_file"
echo "Stage 1b pilot pid=$! physical_gpu=2 log=$log"
