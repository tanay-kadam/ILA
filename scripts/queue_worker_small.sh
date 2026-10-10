#!/usr/bin/env bash
# Memory-capped worker for GPUs shared with other jobs. Same job format and atomic claims as
# queue_worker.sh (so it never duplicates a run), but every train/eval process gets
#   ILA_GPU_MEM_GB   hard per-process memory cap (the run OOMs alone instead of crowding others)
#   EXTRA_SET        config overrides (smaller batches) applied via --set
# Usage: ILA_GPU_MEM_GB=12 EXTRA_SET="train.per_device_batch_size=2 ..." CUDA_VISIBLE_DEVICES=3 \
#        bash scripts/queue_worker_small.sh jobs.txt queue_dir
set -uo pipefail
cd "$(dirname "$0")/.."
JOBS="$1"; QDIR="$2"
mkdir -p "$QDIR/claims" "$QDIR/failed"
export HF_HUB_DISABLE_PROGRESS_BARS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
read -ra OVR <<< "${EXTRA_SET:-}"
SETARGS=(); [[ ${#OVR[@]} -gt 0 ]] && SETARGS=(--set "${OVR[@]}")

while read -r cfg style seed; do
  [[ -z "${cfg:-}" ]] && continue
  id="$(basename "$cfg" .yaml)_${style}_seed${seed}"
  mkdir "$QDIR/claims/$id" 2>/dev/null || continue
  res=$($PY -c "import sys; sys.path.insert(0,'src'); from ila.config import load_config; print(load_config('$cfg')['results_dir'])")
  mkdir -p "$res/logs"
  echo "[$(date +%T)] gpu=${CUDA_VISIBLE_DEVICES:-?} cap=${ILA_GPU_MEM_GB:-none}GB start $id"
  ok=1
  $PY scripts/train_sft.py --config "$cfg" --style "$style" --seed "$seed" "${SETARGS[@]}" > "$res/logs/train_${style}_seed${seed}.log" 2>&1 || ok=0
  if [[ $ok == 1 ]]; then
    for es in gsm8k_test math500; do
      $PY scripts/evaluate.py --config "$cfg" --style "$style" --seed "$seed" --eval-set "$es" "${SETARGS[@]}" \
        > "$res/logs/eval_${style}_seed${seed}_${es}.log" 2>&1 || ok=0
    done
  fi
  if [[ $ok == 1 ]]; then echo "[$(date +%T)] gpu=${CUDA_VISIBLE_DEVICES:-?} done  $id"
  else echo "[$(date +%T)] gpu=${CUDA_VISIBLE_DEVICES:-?} FAILED $id"; touch "$QDIR/failed/$id"; fi
done < "$JOBS"
echo "[$(date +%T)] gpu=${CUDA_VISIBLE_DEVICES:-?} small worker finished"
