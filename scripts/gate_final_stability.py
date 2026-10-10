"""Check baseline-relative quality from raw outputs and summarize the final attempt."""
from __future__ import annotations
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ila.config import load_config
from ila.textio import read_jsonl
from diagnose_stage1_outputs import loop_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1c_final_stability.yaml")
    ap.add_argument("--phase", choices=("engineering", "confirmation", "report"), required=True)
    ap.add_argument("--style", default="armA_random")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config); root = Path(cfg["results_dir"])
    if args.phase == "report":
        gates = [json.loads(p.read_text()) for p in sorted(root.glob("quality_*.json"))]
        if not gates:
            raise SystemExit("no recorded gates; cannot summarize")
        analysis_path = root / "analysis_v2.json"
        analysis = json.loads(analysis_path.read_text()) if analysis_path.exists() else None
        failed = [g for g in gates if not g["pass"]]
        if not failed and (analysis is None or len(gates) != 12):
            raise SystemExit("incomplete experiment; no terminal result")
        decision = "STOP_QUALITY_FAILURE" if failed else analysis["decision"]
        report = {"decision": decision, "gates": gates, "analysis": analysis,
                  "maximum_training_runs": 9, "next_action": "shelve_original_usable_cost_claim" if decision.startswith("STOP") else "review_stable_composition_signal",
                  "benchmark_caveat": "previously observed engineering and confirmation examples"}
        (root / "final_report.json").write_text(json.dumps(report, indent=2) + "\n")
        (root / "final_report.md").write_text(f"# Final stability attempt\n\nDecision: {decision}\n\nRecorded quality gates: {len(gates)}. Failed gates: {len(failed)}.\n\nNext action: {report['next_action']}.\n\nFull details and raw provenance are in final_report.json. Benchmarks were previously observed.\n")
        print(decision); return
    sets = cfg["stabilization"]["engineering_sets" if args.phase == "engineering" else "confirmation_sets"]
    checks = {}
    for name in sets:
        target_rows = read_jsonl(Path(cfg["data_dir"]) / cfg["eval"]["sets"][name]["file"])
        eval_name = name if args.phase == "engineering" else ("gsm8k_test" if name.startswith("gsm8k") else "math500")
        raw = read_jsonl(root / args.style / f"seed{args.seed}" / "eval" / eval_name / "outputs.jsonl")
        lookup = {r["uid"]: r for r in raw}
        if len(lookup) != len(raw):
            raise SystemExit("duplicate generated UIDs")
        candidate = [lookup[r["uid"]] for r in target_rows]
        base = read_jsonl(root / "_base/seed0/eval" / name / "outputs.jsonl")
        if [(r["uid"], r["gold"]) for r in candidate] != [(r["uid"], r["gold"]) for r in base]:
            raise SystemExit("base/candidate UID or gold mismatch")
        accuracy = statistics.mean(r["correct"] for r in candidate)
        base_accuracy = statistics.mean(r["correct"] for r in base)
        truncation = statistics.mean(r["truncated"] for r in candidate)
        loops = statistics.mean(loop_metrics(r["output"])["likely_loop"] for r in candidate)
        checks[name] = {"n": len(candidate), "accuracy": accuracy, "base_accuracy": base_accuracy,
                        "accuracy_drop": base_accuracy - accuracy, "truncation_rate": truncation,
                        "likely_loop_rate": loops, "pass": base_accuracy - accuracy <= .05 and truncation < .02 and loops <= .02}
    report = {"style": args.style, "seed": args.seed, "phase": args.phase, "sets": checks,
              "pass": all(c["pass"] for c in checks.values())}
    (root / f"quality_{args.phase}_{args.style}_seed{args.seed}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
