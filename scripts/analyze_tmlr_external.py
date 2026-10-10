"""Verify and analyze the fixed external evaluation matrix without recipe selection."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from ila.textio import read_jsonl,extract_answer,answers_match,answers_match_choice
from ila.cost_boundary import measures,decompose,bootstrap
from diagnose_stage1_outputs import loop_metrics


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--family",choices=("qwen","mistral"),required=True);ap.add_argument("--draws",type=int,default=2000)
    ap.add_argument("--root",type=Path);ap.add_argument("--baseline-root",type=Path)
    ap.add_argument("--recipes",nargs="+",choices=("lr1e5_step15","lr1e5_step60","lr3e6_step60"),default=["lr1e5_step15","lr1e5_step60","lr3e6_step60"])
    ap.add_argument("--eval-sets",nargs="+",choices=("svamp","aqua"),default=["svamp","aqua"]);args=ap.parse_args()
    root=args.root or ROOT/f"results/tmlr_external_{args.family}"
    report={"family":args.family,"selection":"all frozen recipes; no outcome-dependent exclusions", "recipes":{},"baseline":{}}
    lines=[f"# TMLR external validation: {args.family}","","Development-held-out datasets; foundation-model pretraining contamination is not excluded.",""]
    sources={es:read_jsonl(ROOT/f"data/tmlr_external/eval_{es}.jsonl") for es in args.eval_sets}
    aqua_source=[json.loads(line) for line in (ROOT/"data/tmlr_external/source_aqua.json").read_text().splitlines() if line.strip()]
    def load(style,seed,es):
        selected_root=args.baseline_root if style=="_base" and args.baseline_root else root
        path=selected_root/style/f"seed{seed}/eval"/es
        rows=read_jsonl(path/"outputs.jsonl");summ=json.loads((path/"summary.json").read_text())
        if [(r["uid"],r["gold"]) for r in rows]!=[(r["uid"],r["gold"]) for r in sources[es]]:
            raise ValueError("eval UID/gold mismatch")
        for r in rows:
            if len(r["generated_token_ids"])!=r["n_gen_tokens"]:raise ValueError("token-ID/count mismatch")
            pred=extract_answer(r["output"],"boxed_choice" if es=="aqua" else "boxed")
            correct=answers_match_choice(pred,r["gold"]) if es=="aqua" else answers_match(pred,r["gold"])
            if pred!=r["pred"] or correct!=r["correct"]:raise ValueError("independent answer-scoring mismatch")
        point=measures([r["n_gen_tokens"] for r in rows],[r["truncated"] for r in rows],[r["correct"] for r in rows])
        for key in ("accuracy","mean_tokens","truncation_rate"):
            if not np.isclose(point[key],summ[key],rtol=0,atol=1e-12):raise ValueError("summary/raw mismatch")
        if summ["prompt_mode"]!="canonical_chat" or summ["max_new_tokens"]!=6144 or not summ.get("evaluation_fingerprint"):
            raise ValueError("evaluation provenance mismatch")
        point["likely_loop_rate"]=float(np.mean([loop_metrics(r["output"])["likely_loop"] for r in rows]))
        if es=="aqua":
            point["valid_boxed_option_letter_rate"]=float(np.mean([any(answers_match_choice(r["pred"],g) for g in "ABCDE") for r in rows]))
            value_matches=[]
            for r,source_row in zip(rows,sources[es]):
                raw=aqua_source[source_row["source_index"]]
                value=raw["options"][ord(raw["correct"])-ord("A")][2:].strip()
                value_matches.append(answers_match(r["pred"],value))
            point["secondary_boxed_option_value_match_rate"]=float(np.mean(value_matches))
            point["secondary_scoring_caveat"]="diagnostic only; not a replacement for the frozen explicit-letter accuracy"
        return rows,point
    for es in sources:
        report["baseline"][es]=load("_base",0,es)[1]
    for recipe in args.recipes:
        sets={}
        lines += [f"## {recipe}",""]
        for es in sources:
            arms={}
            for arm in ("armA_high","armA_low","armA_random"):
                loaded=[load(f"{recipe}__{arm}",seed,es) for seed in range(3)]
                arrays={"tokens":np.array([[r["n_gen_tokens"] for r in rows] for rows,_ in loaded]),
                        "truncated":np.array([[r["truncated"] for r in rows] for rows,_ in loaded]),
                        "correct":np.array([[r["correct"] for r in rows] for rows,_ in loaded])}
                summary=measures(arrays["tokens"].ravel(),arrays["truncated"].ravel(),arrays["correct"].ravel())
                summary["per_seed"]=[point for _,point in loaded]
                summary["likely_loop_rate"]=float(np.mean([point["likely_loop_rate"] for _,point in loaded]))
                base=report["baseline"][es]["accuracy"]
                summary["quality_pass_all_seeds"]=all(base-p["accuracy"]<=.05 and p["truncation_rate"]<.02 and p["likely_loop_rate"]<=.02 for _,p in loaded)
                arms[arm]={"arrays":arrays,"summary":summary}
            contrasts={}
            for arm in ("armA_high","armA_low"):
                a,b=arms[arm],arms["armA_random"]
                result=decompose(a["summary"],b["summary"])
                result["relative_delta"]=result["delta_tokens"]/b["summary"]["mean_tokens"]
                result["bootstrap"]=bootstrap(a["arrays"],b["arrays"],args.draws)
                result["paired_seed_sensitivity"]=bootstrap(a["arrays"],b["arrays"],args.draws,paired=True)
                result["cap_sensitivity_delta_tokens"]={str(c):float(np.minimum(a["arrays"]["tokens"],c).mean()-np.minimum(b["arrays"]["tokens"],c).mean()) for c in (1024,2048,4096,6144)}
                contrasts[arm+"_vs_random"]=result
                lo,hi=result["bootstrap"]["delta_tokens"]["ci95"]
                lines += [f"- {es}, {arm} vs random: {result['delta_tokens']:+.2f} tokens ({100*result['relative_delta']:+.2f}%), CI [{lo:+.2f}, {hi:+.2f}]."]
            sets[es]={"arms":{k:v["summary"] for k,v in arms.items()},"contrasts":contrasts}
        report["recipes"][recipe]=sets;lines += [""]
    report["integrity"]="PASS"
    (root/"analysis.json").write_text(json.dumps(report,indent=2)+"\n")
    (root/"analysis.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__=="__main__":main()
