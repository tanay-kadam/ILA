#!/usr/bin/env bash
# Gated pipeline v2: base eval (already running on the first GPU) and the pilot training run in parallel
# on two GPUs. Gates are checked afterwards; any failure stops before the full matrix.
set -uo pipefail
cd "$(dirname "$0")/.."
GPUS=($1); CFG=configs/stage1_distill.yaml; R=results/stage1_distill; ST=$R/PIPELINE_STATUS
export HF_HUB_OFFLINE=1 EVAL_PY=.venv-vllm/bin/python ILA_VLLM_UTIL=0.45
say() { echo "[$(date +%F' '%T)] $*" | tee -a $R/pipeline.log; echo "$*" > $ST; }

Q=$R/queue_pilot; mkdir -p $Q; echo "$CFG armA_random 0" > $Q/jobs.txt
say RUNNING_PILOT_AND_BASE_PARALLEL
CUDA_VISIBLE_DEVICES=${GPUS[1]} bash scripts/queue_worker.sh $Q/jobs.txt $Q >> $R/pipeline.log 2>&1 &
PILOT_PID=$!
while [[ ! -f $R/base.flag ]]; do sleep 30; done
ok=$(.venv/bin/python - <<PY
import json
r=[json.load(open("$R/_base/seed0/eval/%s/summary.json"%s))["truncation_rate"] for s in ("gsm8k_test","math500")]
print("1" if all(x<0.20 for x in r) else "0", r)
PY
)
echo "base truncation: $ok" | tee -a $R/pipeline.log
if [[ ${ok:0:1} != 1 ]]; then kill $PILOT_PID 2>/dev/null; pkill -P $PILOT_PID 2>/dev/null; say STOPPED_BASE_TRUNCATION_GATE; exit 1; fi

say WAIT_PILOT
wait $PILOT_PID
v=$(.venv/bin/python scripts/analyze_stage1.py --config $CFG --pilot 2>&1 | tee -a $R/pipeline.log | grep '"verdict"')
echo "$v" | grep -q PASS || { say "STOPPED_PILOT_GATE $v"; exit 1; }

Q=$R/queue_full; mkdir -p $Q; : > $Q/jobs.txt
echo "$CFG armA_random 0" >> $Q/jobs.txt
for s in armA_high armA_low; do echo "$CFG $s 0" >> $Q/jobs.txt; done
for sd in 1 2; do for n in armA_random armA_high armA_low; do echo "$CFG $n $sd" >> $Q/jobs.txt; done; done
for j in $(seq -w 0 15); do echo "$CFG b$j 0" >> $Q/jobs.txt; done
for sd in 1 2; do for n in b00 b01; do echo "$CFG $n $sd" >> $Q/jobs.txt; done; done
say RUNNING_FULL
for g in "${GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=$g bash scripts/queue_worker.sh $Q/jobs.txt $Q >> $R/worker_gpu$g.log 2>&1 &
done
wait
say FULL_DONE
