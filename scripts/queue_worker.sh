#!/usr/bin/env bash
# One worker bound to one GPU. Claims jobs from a shared list (atomic mkdir claim), so
# faster workers take more jobs. Each job = train + eval on both sets for (config,style,seed).
# Usage: CUDA_VISIBLE_DEVICES=1 bash scripts/queue_worker.sh jobs.txt queue_dir
# jobs.txt lines: <config> <style> <seed>
set -uo pipefail
cd "$(dirname "$0")/.."
JOBS="$1"; QDIR="$2"
mkdir -p "$QDIR/claims" "$QDIR/failed"
export HF_HUB_DISABLE_PROGRESS_BARS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
EPY="${EVAL_PY:-$PY}"

while read -r cfg style seed; do
  [[ -z "${cfg:-}" ]] && continue
  id="$(basename "$cfg" .yaml)_${style}_seed${seed}"
  mkdir "$QDIR/claims/$id" 2>/dev/null || continue
  res=$($PY -c "import sys; sys.path.insert(0,'src'); from ila.config import load_config; print(load_config('$cfg')['results_dir'])")
  mkdir -p "$res/logs"
  echo "[$(date +%T)] gpu=${CUDA_VISIBLE_DEVICES:-?} start $id"
  ok=1
  $PY scripts/train_sft.py --config "$cfg" --style "$style" --seed "$seed" > "$res/logs/train_${style}_seed${seed}.log" 2>&1 || ok=0
  if [[ $ok == 1 ]]; then
    for es in gsm8k_test math500; do
      $EPY scripts/evaluate.py --config "$cfg" --style "$style" --seed "$seed" --eval-set "$es" \
        > "$res/logs/eval_${style}_seed${seed}_${es}.log" 2>&1 || ok=0
    done
  fi
  if [[ $ok == 1 ]]; then echo "[$(date +%T)] gpu=${CUDA_VISIBLE_DEVICES:-?} done  $id"
  else echo "[$(date +%T)] gpu=${CUDA_VISIBLE_DEVICES:-?} FAILED $id"; touch "$QDIR/failed/$id"; fi
done < "$JOBS"
echo "[$(date +%T)] gpu=${CUDA_VISIBLE_DEVICES:-?} worker finished"
