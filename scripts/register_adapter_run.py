"""Register an existing LoRA adapter as an evaluation-only run."""

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
    ap.add_argument("--style", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adapter", type=Path, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if not (args.adapter / "adapter_config.json").is_file():
        raise SystemExit(f"not a LoRA adapter directory: {args.adapter}")
    run_dir = Path(cfg["results_dir"]) / args.style / f"seed{args.seed}"
    if run_dir.exists():
        raise SystemExit(f"refusing to overwrite adapter run: {run_dir}")
    run_dir.mkdir(parents=True)
    record = {
        "status": "completed",
        "kind": "evaluation_only_adapter",
        "model_dir": cfg["model"],
        "adapter_dir": str(args.adapter),
        "max_lora_rank": int(cfg["train"]["lora_r"]),
        "seed": args.seed,
        "config": cfg,
    }
    (run_dir / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

