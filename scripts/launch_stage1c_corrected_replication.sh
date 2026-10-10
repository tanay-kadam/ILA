#!/usr/bin/env bash
# Corrected Stage 1c seed-1 replication. Never launches the Arm A/B matrix.
set -euo pipefail

cd "$(dirname "$0")/.."

gpu="${ILA_STAGE1C_GPU:-2}"
config="configs/stage1c_q7_lr1e5_replfix.yaml"
results="results/stage1c_q7_lr1e5_replfix"
controller="results/stage1c_replfix_controller"
log_dir="logs/stage1c_replfix"
log="$log_dir/pipeline.log"
pid_file="$log_dir/pipeline.pid"
status_file="$controller/STAGE1C_REPLFIX_STATUS"

set_status() {
  mkdir -p "$controller"
  echo "$1 $(date -Is)" | tee "$status_file"
}

run_worker() {
  echo "STARTED $(date -Is) host=$(hostname) physical_gpu=$gpu"
  set_status "RUNNING_PRECHECK"
  test -f docs/stage1c_corrected_replication.md
  test -f data/stage1/train_armA_random.jsonl
  test -f results/stage1c_q7_lr1e5/checkpoint_step015/seed0/adapter_config.json \
    || test -f results/stage1c_q7_lr1e5/armA_random/seed0/trajectory_adapters/step015/adapter_config.json
  if [[ -e "$results" || -e results/stage1c_replfix_smoke ]]; then
    echo "Corrected replication output already exists; refusing to overwrite" >&2
    return 10
  fi

  PYTHONPATH=src .venv/bin/python scripts/materialize_stage1c_splits.py \
    --config "$config" --source-results results/stage1_q7_chat

  set_status "RUNNING_SMOKE"
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 PYTHONPATH=src \
    .venv/bin/python scripts/train_sft.py \
      --config "$config" --style armA_random --seed 1 --smoke-only \
      --set results_dir=results/stage1c_replfix_smoke train.max_steps=1 'train.snapshot_steps=[1]'

  set_status "SMOKE_PASSED_RUNNING_SEED1_230_STEP_SCHEDULE"
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 PYTHONPATH=src \
    .venv/bin/python scripts/train_sft.py \
      --config "$config" --style armA_random --seed 1

  PYTHONPATH=src .venv/bin/python scripts/register_stage1c_checkpoints.py \
    --config "$config" --seed 1

  for eval_set in gsm8k_confirm400 math500_confirm400; do
    CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 ILA_VLLM_UTIL=0.85 PYTHONPATH=src \
      .venv-vllm/bin/python scripts/evaluate.py \
        --config "$config" --style checkpoint_step015 --seed 1 --eval-set "$eval_set"
  done

  HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/diagnose_stage1_outputs.py \
    --set gsm8k_confirm400="$results/checkpoint_step015/seed1/eval/gsm8k_confirm400/outputs.jsonl" \
    --set math500_confirm400="$results/checkpoint_step015/seed1/eval/math500_confirm400/outputs.jsonl" \
    --tokenizer Qwen/Qwen2.5-7B-Instruct \
    --output "$results/checkpoint_step015/seed1/diagnostic"

  if PYTHONPATH=src .venv/bin/python scripts/select_stage1c_checkpoint.py \
      --config "$config" --mode confirmation --style checkpoint_step015 --seed 1; then
    set_status "COMPLETED_CORRECTED_REPLICATION_PASS_PENDING_MATRIX_APPROVAL"
  else
    set_status "COMPLETED_CORRECTED_REPLICATION_FAIL_STOP_STAGE1C"
  fi
}

if [[ "${1:-}" == "--worker" ]]; then
  trap 'code=$?; if [[ $code -ne 0 ]]; then set_status "FAILED_EXIT_$code"; fi' EXIT
  run_worker
  exit 0
fi

mkdir -p "$log_dir" "$controller"
if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  echo "Corrected replication is already running as PID $(cat "$pid_file")"
  exit 1
fi
nohup "$0" --worker >"$log" 2>&1 < /dev/null &
echo "$!" >"$pid_file"
echo "Stage 1c corrected replication pid=$! physical_gpu=$gpu log=$log status=$status_file"

