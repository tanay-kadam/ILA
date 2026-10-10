"""Launch both bounded read-only diagnostics in independent server sessions."""
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/"results/tmlr_eos_effectiveness_probe"
if out.exists():raise SystemExit("existing diagnostic preserved")
out.mkdir(parents=True)
records=[]
for family in ("qwen","mistral"):
    command=[str(ROOT/".venv/bin/python"),"-u",str(ROOT/"scripts/probe_terminal_eos.py"),"--family",family]
    with (out/f"{family}.log").open("w") as stream:
        p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    records.append({"family":family,"pid":p.pid,"command":command,"status":"LAUNCHED_READONLY_PROBE_NOT_TRAINING"})
(out/"launch.json").write_text(json.dumps(records,indent=2)+"\n")
print(json.dumps(records))
