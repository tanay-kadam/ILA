"""Registered analysis for the three-condition, three-seed Stage 1c dose-60 gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from analyze_gate0 import contrast  # noqa: E402
from analyze_stage1 import load_run, stack  # noqa: E402
from ila.config import load_config  # noqa: E402
from ila.dose_gate import contrast_gate, outcome  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1c_dose60.yaml")
    ap.add_argument("--output-name", default="analysis_v2", help="separate output preserves historical analysis")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dose = int(cfg["train"]["stop_after_steps"])
    root = Path(cfg["results_dir"])
    rng = np.random.default_rng(cfg["analysis"]["boot_seed"])
    names = ("armA_high", "armA_low", "armA_random")
    runs, missing = {}, []
    for name in names:
        for seed in (0, 1, 2):
            run = load_run(root, name, seed)
            if run is None:
                missing.append(f"{name}/seed{seed}")
            else:
                runs[(name, seed)] = run
    if missing:
        raise SystemExit(f"incomplete dose-60 matrix: {missing}")
    reference_uids = next(iter(runs.values()))["uids"]
    run_metrics = {}
    for (name, seed), run in runs.items():
        if run["uids"] != reference_uids or len(reference_uids) != 1000:
            raise SystemExit(f"evaluation UID/count mismatch: {name}/seed{seed}")
        record = json.loads((root / name / f"seed{seed}" / "run_record.json").read_text())
        if record.get("status") != "completed":
            raise SystemExit(f"incomplete run record: {name}/seed{seed}")
        if record.get("kind") != "evaluation_only_adapter":
            if record.get("global_steps") != dose or not record.get("device", "").startswith("cuda"):
                raise SystemExit(f"training dose/device mismatch: {name}/seed{seed}")
        elif not str(record.get("adapter_dir", "")).endswith(f"/step{dose:03d}"):
            raise SystemExit(f"reused checkpoint dose mismatch: {name}/seed{seed}")
        if not Path(record["adapter_dir"], "adapter_config.json").is_file():
            raise SystemExit(f"missing adapter: {name}/seed{seed}")
        for es, summary in run["summary"].items():
            for key, expected in {"n": 500, "prompt_mode": "canonical_chat", "decoding": "greedy", "backend": "vllm", "max_new_tokens": 6144}.items():
                if summary.get(key) != expected:
                    raise SystemExit(f"evaluation provenance mismatch: {name}/seed{seed}/{es}/{key}")
        for key, array in (("mean_tokens", "tok"), ("accuracy", "cor"), ("truncation_rate", "trunc")):
            raw = float(run[array].mean())
            reported = float(np.mean([s[key] for s in run["summary"].values()]))
            if not np.isclose(raw, reported, rtol=0, atol=1e-12):
                raise SystemExit(f"raw/summary mismatch: {name}/seed{seed}/{key}")
        run_metrics[f"{name}/seed{seed}"] = {
            "mean_tokens": float(run["tok"].mean()),
            "accuracy": float(run["cor"].mean()),
            "truncation_rate": float(run["trunc"].mean()),
            "benchmarks": run["summary"],
        }
    arms = {name: stack([runs[(name, seed)] for seed in (0, 1, 2)]) for name in names}
    for arm in arms.values():
        arm["seeds"] = [0, 1, 2]
    comparisons = {}
    continue_names = []
    dose15_means = {}
    baseline_checks = {}
    for (name, seed), run in runs.items():
        checks = {}
        for es, cand in run["summary"].items():
            base_dir = root / "_base/seed0/eval" / es
            base = json.loads((base_dir / "summary.json").read_text())
            base_run = load_run(root, "_base", 0)
            if base_run is None or base_run["uids"] != reference_uids:
                raise SystemExit("baseline evaluation is missing or misaligned")
            checks[es] = {
                "accuracy_drop_from_base": base["accuracy"] - cand["accuracy"],
                "accuracy_pass": base["accuracy"] - cand["accuracy"] <= 0.05,
                "truncation_pass": cand["truncation_rate"] < 0.02,
            }
        baseline_checks[f"{name}/seed{seed}"] = checks
    for name in names if cfg.get("compare_dose15", True) else ():
        prior = [load_run(Path("results/stage1c_matrix"), name, seed) for seed in (0, 1, 2)]
        if any(run is None or run["uids"] != reference_uids for run in prior):
            raise SystemExit(f"missing or misaligned step-15 comparison: {name}")
        dose15_means[name] = float(np.mean([run["tok"].mean() for run in prior]))
    for name in ("armA_high", "armA_low"):
        result = contrast(arms[name], arms["armA_random"], cfg["analysis"]["n_boot"], rng)
        preferred = name if result["delta_tokens"] < 0 else "armA_random"
        other = "armA_random" if preferred == name else name
        accuracy_loss = float(arms[other]["cor"].mean() - arms[preferred]["cor"].mean())
        max_trunc = float(max(arms[name]["trunc"].mean(), arms["armA_random"]["trunc"].mean()))
        baseline_pass = all(c["accuracy_pass"] and c["truncation_pass"]
                            for arm in (name, "armA_random") for seed in range(3)
                            for c in baseline_checks[f"{arm}/seed{seed}"].values())
        result["registered_gate"] = contrast_gate(result, accuracy_loss, max_trunc, baseline_pass)
        result["registered_gate"]["preferred_condition"] = preferred
        proceed = result["registered_gate"]["continue_to_step230"]
        result["registered_gate"]["continue_to_step230"] = proceed and dose == 60 and cfg.get("allow_step230", True)
        comparisons[f"{name}_vs_armA_random"] = result
        if dose15_means:
            result["dose15_delta_tokens"] = dose15_means[name] - dose15_means["armA_random"]
            result["contrast_change_from_dose15_tokens"] = result["delta_tokens"] - result["dose15_delta_tokens"]
        if proceed:
            continue_names.append(name)
    report = {
        "experiment": cfg["experiment"],
        "dose": dose,
        "n_runs": 9,
        "integrity": "PASS",
        "runs": run_metrics,
        "dose15_mean_tokens": dose15_means,
        "baseline_checks": baseline_checks,
        "analysis_revision": "audit_fix_v2",
        "baseline_gate_is_prospective_for_new_runs": True,
        "comparisons": comparisons,
        "decision": outcome(r["registered_gate"] for r in comparisons.values()),
        "step230_candidate_conditions": continue_names if dose == 60 and cfg.get("allow_step230", True) else [],
        "note": "This script never launches step 230.",
    }
    (root / f"{args.output_name}.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [f"# Stage 1c dose-{dose} analysis", "", f"Decision: {report['decision']}",
             "", "Nine runs; raw metrics, evaluation alignment, adapter provenance, and training dose verified.", ""]
    for name, result in comparisons.items():
        lo, hi = result["ci95_hierarchical"]
        lines += [f"## {name}", "", f"Mean token difference: {result['delta_tokens']:+.2f} "
                  f"({100 * result['rel_delta_tokens']:+.2f}%). Hierarchical 95% CI: [{lo:+.2f}, {hi:+.2f}].", "",
                  f"Accuracy difference: {100 * result['delta_accuracy']:+.2f} percentage points. "
                  f"Seed signs consistent: {result['sign_consistent_across_seeds']}. "
                  f"Step-230 gate: {result['registered_gate']['continue_to_step230']}.", ""]
        if dose15_means:
            lines += [f"Step-15 token difference: {result['dose15_delta_tokens']:+.2f}; "
                      f"change in contrast from step 15 to {dose}: {result['contrast_change_from_dose15_tokens']:+.2f} tokens "
                      "(descriptive comparison).", ""]
    lines += ["Signals and quality failures are reported separately. Historical results are preserved; baseline checks added in v2 are retrospective for older runs.", ""]
    (root / f"{args.output_name}.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
