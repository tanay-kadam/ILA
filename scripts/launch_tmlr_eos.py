"""Start the fresh EOS controller in an independent server session."""
import json
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/"results/tmlr_eos_control_v3").exists() or (ROOT/"results/tmlr_eos_launch_v3.json").exists():
    raise SystemExit("existing EOS launch/artifacts preserved; inspect before retry")
command=[str(ROOT/".venv/bin/python"),"-u",str(ROOT/"scripts/tmlr_eos_control.py")]
log=ROOT/"results/tmlr_eos_control_v3_launch.log"
with log.open("w") as stream:
    process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,
        stderr=subprocess.STDOUT,start_new_session=True)
record={"pid":process.pid,"command":command,"log":str(log),"started":time.strftime("%Y-%m-%dT%H:%M:%S"),"status":"launched_not_completed"}
(ROOT/"results/tmlr_eos_launch_v3.json").write_text(json.dumps(record,indent=2))
print(json.dumps(record))
