#!/usr/bin/env bash
# Stage 1 launcher. One detached worker per GPU you pass; no GPU is chosen for you.
#   bash scripts/launch_stage1.sh pilot 3          # base-model eval, then the pilot run (armA_random seed 0)
#   bash scripts/launch_stage1.sh full "3 5 6"     # all 29 runs (completed runs are skipped)
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:?pilot|full}"; read -ra GPUS <<< "${2:?GPU list required, e.g. \"3\" or \"3 5 6\"}"
CFG=configs/stage1.yaml
Q=results/queue_stage1_${MODE}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$Q" results/stage1/logs
export HF_HUB_OFFLINE=1   # models are cached; the shared server IP is rate limited by the Hub
export EVAL_PY=.venv-vllm/bin/python ILA_VLLM_UTIL="${ILA_VLLM_UTIL:-0.45}"
: > "$Q/jobs.txt"

if [[ "$MODE" == pilot ]]; then
  echo "$CFG armA_random 0" >> "$Q/jobs.txt"
else
  echo "$CFG armA_random 0" >> "$Q/jobs.txt"
  for s in armA_high armA_low; do echo "$CFG $s 0" >> "$Q/jobs.txt"; done
  for s in 1 2; do for n in armA_random armA_high armA_low; do echo "$CFG $n $s" >> "$Q/jobs.txt"; done; done
  for j in $(seq -w 0 15); do echo "$CFG b$j 0" >> "$Q/jobs.txt"; done
  for s in 1 2; do for n in b00 b01; do echo "$CFG $n $s" >> "$Q/jobs.txt"; done; done
fi
echo "$(wc -l < "$Q/jobs.txt") jobs -> $Q"

if [[ "$MODE" == pilot ]]; then
  g="${GPUS[0]}"
  CUDA_VISIBLE_DEVICES=$g setsid nohup bash -c "
    for es in gsm8k_test math500; do
      \$EVAL_PY scripts/evaluate.py --config $CFG --style _base --seed 0 --base --eval-set \$es \
        > results/stage1/logs/eval_base_\$es.log 2>&1
    done
    bash scripts/queue_worker.sh $Q/jobs.txt $Q
  " > "$Q/worker_gpu${g}.log" 2>&1 < /dev/null &
  echo "pilot worker GPU $g pid $!"
else
  for g in "${GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=$g setsid nohup bash scripts/queue_worker.sh "$Q/jobs.txt" "$Q" \
      > "$Q/worker_gpu${g}.log" 2>&1 < /dev/null &
    echo "worker GPU $g pid $!"
  done
fi
echo "$Q" > results/LATEST_QUEUE
