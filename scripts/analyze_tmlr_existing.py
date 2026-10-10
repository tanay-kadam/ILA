"""Reanalyze existing raw generations into separate TMLR boundary artifacts."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ila.textio import read_jsonl
from ila.cost_boundary import measures, decompose, bootstrap
from diagnose_stage1_outputs import loop_metrics

RECIPES = {"lr1e5_step15": "stage1c_matrix", "lr1e5_step60": "stage1c_dose60", "lr3e6_step60": "stage1c_final_stability"}
ARMS = ("armA_high", "armA_random", "armA_low")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=2000); args = ap.parse_args()
    out = ROOT / "results/tmlr_existing_analysis"; out.mkdir(exist_ok=True)
    report = {"analysis_type": "post_hoc_descriptive_and_sensitivity", "bootstrap_draws": args.draws,
              "seed": 20261007, "selection_caveat": "finished cohorts differ; components are not causal mediation", "recipes": {}}
    lines = ["# TMLR existing-output boundary analysis", "", "Post-hoc descriptive analysis; original artifacts preserved.", ""]
    for recipe, folder in RECIPES.items():
        by_set = {}
        all_rows = {}
        for arm in ARMS:
            all_rows[arm] = {}
            for es in ("gsm8k_test", "math500"):
                all_rows[arm][es] = [read_jsonl(ROOT / "results" / folder / arm / f"seed{s}/eval" / es / "outputs.jsonl") for s in range(3)]
        for es in ("gsm8k_test", "math500", "pooled"):
            arms = {}
            for arm in ARMS:
                rows = all_rows[arm][es] if es != "pooled" else [all_rows[arm]["gsm8k_test"][s] + all_rows[arm]["math500"][s] for s in range(3)]
                ids = [(r["uid"], r["gold"]) for r in rows[0]]
                if any([(r["uid"], r["gold"]) for r in rr] != ids for rr in rows):
                    raise ValueError("seed UID/gold mismatch")
                arrays = {"tokens": np.array([[r["n_gen_tokens"] for r in rr] for rr in rows]),
                          "truncated": np.array([[r["truncated"] for r in rr] for rr in rows]),
                          "correct": np.array([[r["correct"] for r in rr] for rr in rows])}
                summary = measures(arrays["tokens"].ravel(), arrays["truncated"].ravel(), arrays["correct"].ravel())
                summary["likely_loop_rate"] = float(np.mean([loop_metrics(r["output"])["likely_loop"] for rr in rows for r in rr]))
                summary["per_seed"] = [measures(arrays["tokens"][s], arrays["truncated"][s], arrays["correct"][s]) for s in range(3)]
                arms[arm] = {"arrays": arrays, "summary": summary, "ids": ids}
            contrasts = {}
            for arm in ("armA_high", "armA_low"):
                a, b = arms[arm], arms["armA_random"]
                if a["ids"] != b["ids"]:
                    raise ValueError("arm UID/gold mismatch")
                point = decompose(a["summary"], b["summary"])
                point["independent_seed_bootstrap"] = bootstrap(a["arrays"], b["arrays"], args.draws)
                point["paired_seed_sensitivity"] = bootstrap(a["arrays"], b["arrays"], args.draws, paired=True)
                point["cap_sensitivity_delta_tokens"] = {str(c): float(np.minimum(a["arrays"]["tokens"],c).mean()-np.minimum(b["arrays"]["tokens"],c).mean()) for c in (1024,2048,4096,6144)}
                common = ~(a["arrays"]["truncated"] | b["arrays"]["truncated"])
                point["common_finished_descriptive"] = {"n_seed_question_pairs": int(common.sum()),
                    "delta_tokens": float((a["arrays"]["tokens"]-b["arrays"]["tokens"])[common].mean()) if common.any() else None}
                contrasts[arm+"_vs_random"] = point
            by_set[es] = {"arms": {k:v["summary"] for k,v in arms.items()}, "contrasts": contrasts}
            if es == "pooled":
                lines += [f"## {recipe}", ""]
                for arm, c in contrasts.items():
                    lines += [f"- {arm}: total {c['delta_tokens']:+.3f} tokens; frequency component {c.get('frequency_component', float('nan')):+.3f}; finished-length component {c.get('finished_length_component', float('nan')):+.3f}."]
                lines += [""]
        report["recipes"][recipe] = by_set
        (out / "analysis_partial.json").write_text(json.dumps(report, indent=2)+"\n")
    (out / "analysis.json").write_text(json.dumps(report, indent=2)+"\n")
    (out / "analysis.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__": main()
