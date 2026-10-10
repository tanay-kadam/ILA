#!/usr/bin/env bash
# Bounded, unattended Stage 1c stabilization pipeline. Never launches the Arm A/B matrix.
set -euo pipefail

cd "$(dirname "$0")/.."

gpu="${ILA_STAGE1C_GPU:-2}"
controller="results/stage1c_controller"
log_dir="logs/stage1c"
log="$log_dir/pipeline.log"
pid_file="$log_dir/pipeline.pid"
status_file="$controller/STAGE1C_STATUS"
primary="configs/stage1c_q7.yaml"
fallback="configs/stage1c_q7_lr1e5.yaml"

cfg_value() {
  PYTHONPATH=src .venv/bin/python -c \
    "from ila.config import load_config; print(load_config('$1')$2)"
}

set_status() {
  mkdir -p "$controller"
  echo "$1 $(date -Is)" | tee "$status_file"
}

diagnose_style() {
  local cfg="$1" style="$2" seed="$3" phase="$4" set_a="$5" set_b="$6"
  local results
  results=$(cfg_value "$cfg" "['results_dir']")
  HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/diagnose_stage1_outputs.py \
    --set "$set_a=$results/$style/seed$seed/eval/$set_a/outputs.jsonl" \
    --set "$set_b=$results/$style/seed$seed/eval/$set_b/outputs.jsonl" \
    --tokenizer Qwen/Qwen2.5-7B-Instruct \
    --output "$results/$style/seed$seed/$phase"
}

evaluate_set() {
  local cfg="$1" style="$2" seed="$3" eval_set="$4"
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 ILA_VLLM_UTIL=0.85 PYTHONPATH=src \
    .venv-vllm/bin/python scripts/evaluate.py \
      --config "$cfg" --style "$style" --seed "$seed" --eval-set "$eval_set"
}

materialize() {
  local cfg="$1"
  local results
  results=$(cfg_value "$cfg" "['results_dir']")
  if [[ -f "$results/split_manifest.json" ]]; then
    echo "SPLITS_ALREADY_MATERIALIZED config=$cfg results=$results"
    return 0
  fi
  HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/materialize_stage1c_splits.py \
    --config "$cfg" --source-results results/stage1_q7_chat
}

run_rate() {
  local cfg="$1" label="$2"
  local results
  results=$(cfg_value "$cfg" "['results_dir']")
  echo "RATE_START label=$label config=$cfg results=$results $(date -Is)"
  materialize "$cfg"

  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 PYTHONPATH=src \
    .venv/bin/python scripts/train_sft.py \
      --config "$cfg" --style armA_random --seed 0
  PYTHONPATH=src .venv/bin/python scripts/register_stage1c_checkpoints.py --config "$cfg"

  for step in 15 30 60 115 230; do
    style=$(printf 'checkpoint_step%03d' "$step")
    evaluate_set "$cfg" "$style" 0 gsm8k_dev100
    evaluate_set "$cfg" "$style" 0 math500_dev100
    diagnose_style "$cfg" "$style" 0 diagnostic gsm8k_dev100 math500_dev100
  done

  if ! PYTHONPATH=src .venv/bin/python scripts/select_stage1c_checkpoint.py \
      --config "$cfg" --mode engineering; then
    echo "RATE_NO_ENGINEERING_PASS label=$label $(date -Is)"
    return 2
  fi
  selected=$(tr -d '[:space:]' < "$results/selected_step.txt")
  style=$(printf 'checkpoint_step%03d' "$selected")
  echo "RATE_SELECTED label=$label step=$selected style=$style"

  evaluate_set "$cfg" "$style" 0 gsm8k_confirm400
  evaluate_set "$cfg" "$style" 0 math500_confirm400
  diagnose_style "$cfg" "$style" 0 diagnostic_confirm gsm8k_confirm400 math500_confirm400
  if ! PYTHONPATH=src .venv/bin/python scripts/select_stage1c_checkpoint.py \
      --config "$cfg" --mode confirmation --style "$style" --seed 0; then
    echo "RATE_CONFIRMATION_FAIL label=$label step=$selected $(date -Is)"
    return 3
  fi

  echo "REPLICATION_START label=$label step=$selected seed=1 $(date -Is)"
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 PYTHONPATH=src \
    .venv/bin/python scripts/train_sft.py \
      --config "$cfg" --style armA_random --seed 1 \
      --set "train.max_steps=$selected" 'train.snapshot_steps=[]'
  evaluate_set "$cfg" armA_random 1 gsm8k_confirm400
  evaluate_set "$cfg" armA_random 1 math500_confirm400
  diagnose_style "$cfg" armA_random 1 diagnostic gsm8k_confirm400 math500_confirm400
  if ! PYTHONPATH=src .venv/bin/python scripts/select_stage1c_checkpoint.py \
      --config "$cfg" --mode confirmation --style armA_random --seed 1; then
    echo "RATE_REPLICATION_FAIL label=$label step=$selected $(date -Is)"
    return 4
  fi
  echo "$label $selected" > "$controller/selected_recipe.txt"
  return 0
}

run_worker() {
  echo "STARTED $(date -Is) host=$(hostname) physical_gpu=$gpu"
  set_status "RUNNING_PRECHECK"
  test -f results/stage1_q7_chat/STAGE1B_PILOT_STATUS
  test -f results/stage1_q7_chat/armA_random/seed0/adapter/adapter_config.json
  test -f data/stage1/train_armA_random.jsonl
  test -f data/stage1/eval_gsm8k_test.jsonl
  test -f data/stage1/eval_math500.jsonl
  if [[ -e results/stage1c_q7 || -e results/stage1c_q7_lr1e5 || -e results/stage1c_smoke ]]; then
    echo "Stage 1c output already exists; refusing to overwrite" >&2
    return 10
  fi

  mkdir -p "$controller"
  HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/diagnose_stage1_outputs.py \
    --set gsm8k=results/stage1_q7_chat/armA_random/seed0/eval/gsm8k_test/outputs.jsonl \
    --set math500=results/stage1_q7_chat/armA_random/seed0/eval/math500/outputs.jsonl \
    --tokenizer Qwen/Qwen2.5-7B-Instruct \
    --output "$controller/prior_pilot_diagnostic"
  HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/audit_stage1_targets.py \
    --config "$primary" --style armA_random --output "$controller/training_target_audit.json"

  set_status "RUNNING_SMOKE"
  materialize "$primary"
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 PYTHONPATH=src \
    .venv/bin/python scripts/train_sft.py \
      --config "$primary" --style armA_random --seed 0 --smoke-only \
      --set results_dir=results/stage1c_smoke train.max_steps=1 'train.snapshot_steps=[1]'
  PYTHONPATH=src .venv/bin/python scripts/register_adapter_run.py \
    --config "$primary" --style adapter_smoke --seed 0 \
    --adapter results/stage1_q7_chat/armA_random/seed0/adapter
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 ILA_VLLM_UTIL=0.85 PYTHONPATH=src \
    .venv-vllm/bin/python scripts/evaluate.py \
      --config "$primary" --style adapter_smoke --seed 0 --eval-set gsm8k_dev100 --limit 2
  set_status "SMOKE_PASSED_RUNNING_LR1E4"

  if run_rate "$primary" "lr1e-4"; then
    set_status "COMPLETED_REPLICATED_PASS_PENDING_MATRIX_APPROVAL"
    return 0
  else
    first_code=$?
  fi
  if [[ "$first_code" -ne 2 ]]; then
    set_status "COMPLETED_STAGE1C_FAIL_LR1E4_CODE_$first_code"
    return 0
  fi

  set_status "LR1E4_NO_PASS_RUNNING_PREREGISTERED_LR1E5"
  if run_rate "$fallback" "lr1e-5"; then
    set_status "COMPLETED_REPLICATED_PASS_PENDING_MATRIX_APPROVAL"
    return 0
  else
    second_code=$?
  fi
  set_status "COMPLETED_STAGE1C_FAIL_LR1E5_CODE_$second_code"
}

if [[ "${1:-}" == "--worker" ]]; then
  trap 'code=$?; if [[ $code -ne 0 ]]; then set_status "FAILED_EXIT_$code"; fi' EXIT
  run_worker
  exit 0
fi

mkdir -p "$log_dir" "$controller"
if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  echo "Stage 1c is already running as PID $(cat "$pid_file")"
  exit 1
fi
nohup "$0" --worker >"$log" 2>&1 < /dev/null &
echo "$!" >"$pid_file"
echo "Stage 1c pid=$! physical_gpu=$gpu log=$log status=$status_file"
