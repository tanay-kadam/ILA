"""Detached CPU-only watcher for the already-running fixed experiment branches."""
import subprocess
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
controller=ROOT/"results/tmlr_strengthening_controller"
while True:
    states={family:(controller/f"{family}_STATUS").read_text() if (controller/f"{family}_STATUS").exists() else "PENDING" for family in ("qwen","mistral")}
    if any(s.startswith("FAILED") for s in states.values()):
        (controller/"combined_STATUS").write_text("INCOMPLETE_BRANCH_REQUIRES_REVIEW\n"+str(states))
        raise SystemExit(2)
    if all(s.startswith("COMPLETED_ANALYSIS_READY") for s in states.values()):break
    time.sleep(30)
subprocess.run([str(ROOT/".venv/bin/python"),"scripts/export_tmlr_strengthening.py"],cwd=ROOT,check=True)
(controller/"combined_STATUS").write_text("COMPLETED_VERIFIED_EVIDENCE_READY_FOR_MANUSCRIPT_REVIEW\n")
