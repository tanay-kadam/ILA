"""Detach the cached-source CPU feasibility check; independent of laptop VPN."""
import json
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]
control=ROOT/"results/tmlr_matching_feasibility_v2"
if control.exists():raise SystemExit("existing feasibility artifacts preserved")
control.mkdir(parents=True)
command=[str(ROOT/".venv/bin/python"),"-u",str(ROOT/"scripts/assess_cached_trace_pairs.py"),"--output",str(control/"cached_pairs.json")]
with (control/"source_inspection.log").open("w") as stream:
    p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
(control/"launch.json").write_text(json.dumps({"pid":p.pid,"command":command,"started":time.strftime("%Y-%m-%dT%H:%M:%S"),"status":"CPU_FEASIBILITY_RUNNING_NO_TRAINING"},indent=2))
print(p.pid)
