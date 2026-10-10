#!/usr/bin/env bash
cd "$(dirname "$0")/.."
Q=$(cat results/LATEST_QUEUE)
echo "queue: $Q"
echo "1.5B evals complete: $(ls results/gate0_1p5b/*/seed*/eval/math500/summary.json 2>/dev/null | wc -l)/15"
echo "paraphrase evals complete: $(ls results/gate0_para/*/seed*/eval/math500/summary.json 2>/dev/null | wc -l)/15"
echo "failed jobs: $(ls "$Q/failed" 2>/dev/null | wc -l)"
ls "$Q/failed" 2>/dev/null
echo "active train/eval processes: $(pgrep -fc 'scripts/(train_sft|evaluate).py')"
grep -h 'FAILED' "$Q"/*.log 2>/dev/null
