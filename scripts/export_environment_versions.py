"""Export package names/versions only; no index URLs, secrets or model updates."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
if args.output.exists():raise ValueError("existing environment record preserved")
packages={d.metadata["Name"]:d.version for d in importlib.metadata.distributions() if d.metadata.get("Name")}
record={"python":sys.version,"platform":platform.platform(),"packages":dict(sorted(packages.items())),
        "scope":"installed version snapshot at consolidation, not a reconstructed historical dependency lock or guarantee of deterministic kernels"}
try:
    import torch
    record["torch_cuda_build"]=torch.version.cuda
except ImportError:pass
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(record,indent=2)+"\n")
print(f"Exported {len(packages)} package versions")
