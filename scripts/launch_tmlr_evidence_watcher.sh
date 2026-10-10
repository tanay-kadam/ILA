#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
controller=results/tmlr_strengthening_controller
if [[ -f "$controller/export.pid" ]] && kill -0 "$(cat "$controller/export.pid")" 2>/dev/null; then
  echo "export watcher already running"; exit 0
fi
nohup .venv/bin/python scripts/await_tmlr_evidence.py > "$controller/evidence_export.log" 2>&1 < /dev/null &
echo "$!" > "$controller/export.pid"
echo "CPU-only evidence watcher pid=$!"
