"""Record scheduler-only pre-full hotfix and reclaim only a proven dead own lease."""
import json
import os
from pathlib import Path
from evaluate import file_hash
ROOT=Path(__file__).resolve().parents[1]
control=ROOT/"results/tmlr_eos_control_v3/controller"
precheck=json.loads((control/"precheck.json").read_text())
record={"kind":"pre-full scheduler safeguard", "reason":"waiting workers must exit when all jobs are claimed, allowing automatic final analysis",
        "scientific_change":"none: loss, data, model, optimizer, evaluation and seed configurations unchanged",
        "precheck_controller_sha256":precheck["code_sha256"]["scripts/tmlr_eos_control.py"],
        "worker_controller_sha256":file_hash(ROOT/"scripts/tmlr_eos_control.py")}
old=ROOT/"results/tmlr_eos_control_v2/controller"
if list((old/"done").glob("*.json")):
    raise RuntimeError("v2 full runs present; manual review required")
old_status=(old/"STATUS").read_text()
record["v2_previous_status"]=old_status
record["v2_status"]="STOPPED_DURING_SMOKES_NO_FULL_RUNS"
(old/"interruption.json").write_text(json.dumps(record,indent=2))
(old/"STATUS").write_text(record["v2_status"]+"\n")
lease=ROOT/"results/tmlr_strengthening_controller/gpu_leases/gpu0"
if lease.exists():
    owner=json.loads((lease/"owner.json").read_text())
    if owner["pid"] != 2105115 or owner["physical_gpu"] != 0:
        raise RuntimeError("lease belongs to another process; untouched")
    try:os.kill(owner["pid"],0)
    except ProcessLookupError:pass
    else:raise RuntimeError("owner alive; lease untouched")
    if {p.name for p in lease.iterdir()} != {"owner.json"}:
        raise RuntimeError("unexpected lease contents; untouched")
    record["reclaimed_dead_own_lease"]=owner
    (lease/"owner.json").unlink()
    lease.rmdir()
(control/"scheduler_fix.json").write_text(json.dumps(record,indent=2))
print(json.dumps(record))
