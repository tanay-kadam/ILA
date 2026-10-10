"""Apply the preregistered Stage 1c engineering or confirmation stability gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ila.config import load_config  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def judge(
    results: Path,
    cfg: dict,
    style: str,
    seed: int,
    sets: list[str],
    diagnostic_dir: str = "diagnostic",
) -> dict:
    gate = cfg["stabilization"]
    diag = load(results / style / f"seed{seed}" / diagnostic_dir / "diagnostic.json")["sets"]
    checks = {}
    for name in sets:
        base = load(results / "_base" / "seed0" / "eval" / name / "summary.json")
        cand = load(results / style / f"seed{seed}" / "eval" / name / "summary.json")
        item = {
            "base_accuracy": base["accuracy"],
            "candidate_accuracy": cand["accuracy"],
            "accuracy_drop_abs": base["accuracy"] - cand["accuracy"],
            "truncation_rate": cand["truncation_rate"],
            "likely_loop_rate": diag[name]["likely_loop_rate"],
        }
        item["accuracy_ok"] = item["accuracy_drop_abs"] <= gate["max_accuracy_drop_abs"]
        item["truncation_ok"] = item["truncation_rate"] <= gate["max_truncation_rate"]
        item["loops_ok"] = item["likely_loop_rate"] <= gate["max_likely_loop_rate"]
        item["pass"] = item["accuracy_ok"] and item["truncation_ok"] and item["loops_ok"]
        checks[name] = item
    return {"style": style, "seed": seed, "sets": checks, "pass": all(x["pass"] for x in checks.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["engineering", "confirmation"], required=True)
    ap.add_argument("--style")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    results = Path(cfg["results_dir"])
    if args.mode == "engineering":
        sets = cfg["stabilization"]["engineering_sets"]
        candidates = [
            judge(results, cfg, f"checkpoint_step{int(step):03d}", 0, sets)
            for step in cfg["stabilization"]["snapshot_steps"]
        ]
        passing = [row for row in candidates if row["pass"]]
        selected = max((int(row["style"].removeprefix("checkpoint_step")) for row in passing), default=None)
        report = {"mode": args.mode, "candidates": candidates, "selected_step": selected, "pass": selected is not None}
        out = results / "engineering_gate.json"
        (results / "selected_step.txt").write_text("" if selected is None else str(selected))
    else:
        if not args.style:
            raise SystemExit("--style is required for confirmation")
        sets = cfg["stabilization"]["confirmation_sets"]
        diagnostic_dir = "diagnostic_confirm" if args.seed == 0 and args.style.startswith("checkpoint_step") else "diagnostic"
        candidate = judge(results, cfg, args.style, args.seed, sets, diagnostic_dir)
        report = {"mode": args.mode, "candidate": candidate, "pass": candidate["pass"]}
        out = results / f"confirmation_gate_{args.style}_seed{args.seed}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
