"""Paired weight10-weight1 comparisons and prespecified pooled cap-hit gate."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from ila.textio import read_jsonl,extract_boxed,answers_match,answers_match_choice
from ila.cost_boundary import bootstrap
from evaluate import file_hash
from tmlr_eos_control import OUT


def arrays(root,arm,es):
    rows=[read_jsonl(root/f"lr1e5_step60__{arm}/seed{s}/eval/{es}/outputs.jsonl") for s in range(3)]
    frozen=read_jsonl(ROOT/f"data/tmlr_external/eval_{es}.jsonl")
    for seed_rows in rows:
        if [(r["uid"],r["gold"]) for r in seed_rows]!=[(r["uid"],r["gold"]) for r in frozen]:
            raise ValueError("UID/gold mismatch")
        for r in seed_rows:
            pred=extract_boxed(r["output"])
            correct=answers_match_choice(pred,r["gold"]) if es=="aqua" else answers_match(pred,r["gold"])
            if pred!=r["pred"] or correct!=r["correct"] or len(r["generated_token_ids"])!=r["n_gen_tokens"]:
                raise ValueError("raw metric mismatch")
    return {k:np.array([[r[field] for r in rs] for rs in rows]) for k,field in
            (("tokens","n_gen_tokens"),("truncated","truncated"),("correct","correct"))}


def main():
    out=OUT/"controller"
    report={"comparison":"terminal EOS weight10 minus fresh matched weight1", "families":{},
            "uncertainty":"paired training seed within arm; shared questions within each benchmark; 2000 draws, RNG20261008",
            "caveats":["exploratory reused evaluation sets", "descriptive decomposition, not causal mediation",
                       "fresh weight1 comparators; historical results not substituted", "three seeds have limited small-effect power",
                       "Mistral original AQuA strict-letter format limitations retained"]}
    arms=("armA_high","armA_low","armA_random")
    for family in ("qwen","mistral"):
        oldroot=OUT/family/"weight1/eval"
        newroot=OUT/family/"weight10/eval"
        old=json.loads((oldroot/"analysis.json").read_text())
        new=json.loads((newroot/"analysis.json").read_text())
        if old["integrity"]!="PASS" or new["integrity"]!="PASS":raise ValueError("missing integrity verification")
        family_report={"arms":{},"benchmark_pooled_accuracy_deltas":{},"source_analysis":{"original":str(oldroot/"analysis.json"),"control":str(newroot/"analysis.json")}}
        data={}
        for es in ("svamp","aqua"):
            data[es]={}
            acc=[]
            for arm in arms:
                a,b=arrays(newroot,arm,es),arrays(oldroot,arm,es)
                data[es][arm]=(a,b)
                contrast=bootstrap(a,b,draws=2000,seed=20261008,paired=True)
                before=old["recipes"]["lr1e5_step60"][es]["arms"][arm]
                after=new["recipes"]["lr1e5_step60"][es]["arms"][arm]
                accuracy_delta=float(a["correct"].mean()-b["correct"].mean());acc.append(accuracy_delta)
                family_report["arms"][f"{es}/{arm}"]={"original":before,"control":after,
                    "paired_bootstrap":contrast,"accuracy_delta":accuracy_delta,
                    "cap_hit_delta":float(a["truncated"].mean()-b["truncated"].mean()),
                    "mean_token_delta":float(a["tokens"].mean()-b["tokens"].mean())}
            family_report["benchmark_pooled_accuracy_deltas"][es]=float(np.mean(acc))
        # Equal benchmark, arm and seed weights; preserve pairing of seeds across
        # both benchmarks within each arm, not across unrelated composition arms.
        rng=np.random.default_rng(20261008);draws=[]
        point=float(np.mean([a["truncated"].mean()-b["truncated"].mean() for bank in data.values() for a,b in bank.values()]))
        for _ in range(2000):
            seeds={arm:rng.integers(0,3,3) for arm in arms}
            values=[]
            for es,bank in data.items():
                n=next(iter(bank.values()))[0]["truncated"].shape[1]
                ix=rng.integers(0,n,n)
                for arm,(a,b) in bank.items():
                    si=seeds[arm]
                    values.append(a["truncated"][si][:,ix].mean()-b["truncated"][si][:,ix].mean())
            draws.append(np.mean(values))
        ci=[float(x) for x in np.quantile(draws,[.025,.975])]
        accuracy_ok=all(v>=-.05 for v in family_report["benchmark_pooled_accuracy_deltas"].values())
        family_report["primary_gate"]={"cap_hit_delta":point,"ci95":ci,"reduction_at_least_10pp":point<=-.10,
            "ci_excludes_zero_in_reduction_direction":ci[1]<0,"accuracy_loss_within_5pp_each_benchmark":accuracy_ok,
            "operational_hypothesis_supported":point<=-.10 and ci[1]<0 and accuracy_ok}
        family_report["analysis_sha256"]={str(root/"analysis.json"):file_hash(root/"analysis.json") for root in (oldroot,newroot)}
        report["families"][family]=family_report
    report["integrity"]="PASS"
    (out/"comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({f:r["primary_gate"] for f,r in report["families"].items()}))


if __name__=="__main__":main()
