"""Register saved adapter snapshots as immutable evaluation runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ila.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    results = Path(cfg["results_dir"])
    source = results / "armA_random" / f"seed{args.seed}"
    train_record = json.loads((source / "run_record.json").read_text())
    manifest = json.loads((source / "trajectory_adapters" / "manifest.json").read_text())
    expected = [int(x) for x in cfg["stabilization"]["snapshot_steps"]]
    if manifest["saved_steps"] != expected:
        raise SystemExit(f"snapshot mismatch: expected {expected}, got {manifest['saved_steps']}")
    for step in expected:
        style = f"checkpoint_step{step:03d}"
        run_dir = results / style / f"seed{args.seed}"
        if run_dir.exists():
            raise SystemExit(f"refusing to overwrite checkpoint record: {run_dir}")
        run_dir.mkdir(parents=True)
        adapter = source / "trajectory_adapters" / f"step{step:03d}"
        record = {
            "status": "completed",
            "kind": "stage1c_adapter_snapshot",
            "source_training_record": str(source / "run_record.json"),
            "optimizer_step": step,
            "seed": args.seed,
            "model_dir": cfg["model"],
            "adapter_dir": str(adapter),
            "max_lora_rank": int(cfg["train"]["lora_r"]),
            "device": train_record["device"],
            "config": cfg,
        }
        (run_dir / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"registered_steps": expected, "results_dir": str(results)}))


if __name__ == "__main__":
    main()

