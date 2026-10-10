#!/usr/bin/env bash
# Full Gate 0 matrix: build data -> (style x seed) train + eval -> analysis.
# Completed runs/evals are skipped, so the script can be re-run after interruptions.
# Usage: bash scripts/run_gate0.sh [config] [extra --set overrides...]
# Env: SEEDS="0" / STYLES="concise verify" restrict the matrix (e.g. one seed per GPU);
#      SKIP_ANALYSIS=1 skips the final analysis (run it once after all shards finish).
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/gate0.yaml}"
shift || true
EXTRA=("$@")

read_list() { python -c "import sys; sys.path.insert(0,'src'); from ila.config import load_config; print(' '.join(map(str, load_config('$CONFIG')['$1'])))"; }
STYLES="${STYLES:-$(read_list styles)}"
SEEDS="${SEEDS:-$(read_list seeds)}"
DATA_DIR=$(python -c "import sys; sys.path.insert(0,'src'); from ila.config import load_config; print(load_config('$CONFIG')['data_dir'])")
RESULTS_DIR=$(python -c "import sys; sys.path.insert(0,'src'); from ila.config import load_config; print(load_config('$CONFIG')['results_dir'])")
mkdir -p "$RESULTS_DIR/logs"

if [[ ! -f "$DATA_DIR/build_report.json" ]]; then
  python scripts/build_data.py --config "$CONFIG" ${EXTRA[@]+--set "${EXTRA[@]}"} 2>&1 | tee "$RESULTS_DIR/logs/build_data.log"
fi

for seed in $SEEDS; do
  for style in $STYLES; do
    tag="${style}_seed${seed}"
    python scripts/train_sft.py --config "$CONFIG" --style "$style" --seed "$seed" \
      ${EXTRA[@]+--set "${EXTRA[@]}"} 2>&1 | tee "$RESULTS_DIR/logs/train_${tag}.log"
    for es in gsm8k_test math500; do
      python scripts/evaluate.py --config "$CONFIG" --style "$style" --seed "$seed" --eval-set "$es" \
        ${EXTRA[@]+--set "${EXTRA[@]}"} 2>&1 | tee "$RESULTS_DIR/logs/eval_${tag}_${es}.log"
    done
  done
done

if [[ "${SKIP_ANALYSIS:-0}" != "1" ]]; then
  python scripts/analyze_gate0.py --config "$CONFIG" ${EXTRA[@]+--set "${EXTRA[@]}"} 2>&1 | tee "$RESULTS_DIR/logs/analysis.log"
fi
