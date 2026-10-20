"""CPU-only matching feasibility from archived banks; no outcome-based selection."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from ila.textio import read_jsonl
from ila.natural import words,length_deciles


def summary(rows):
    lens=np.array([r["n_tokens"] for r in rows],float)
    return {"n":len(rows),"trace_tokens_mean":float(lens.mean()),"trace_tokens_std":float(lens.std()),
            "quantiles":np.quantile(lens,[0,.1,.5,.9,1]).tolist(),
            "reflection_density_mean":float(np.mean([r["reflection_density"] for r in rows])),
            "source_type_counts":dict(Counter(f"{r['source']}|{r['problem_type']}" for r in rows))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    pool=read_jsonl(ROOT/"data/stage1/pool.jsonl")
    groups=defaultdict(list)
    for r in pool:groups[tuple(words(r["question"]))].append(r)
    counts=Counter(len(v) for v in groups.values())
    pairs=[v for v in groups.values() if len({r["target"] for r in v})>=2]
    arms={name:read_jsonl(ROOT/f"data/stage1/train_armA_{name}.jsonl") for name in ("high","low","random")}
    stats={name:summary(rows) for name,rows in arms.items()}
    contrasts={}
    for name in ("high","low"):
        a,b=stats[name],stats["random"]
        sd=np.sqrt((a["trace_tokens_std"]**2+b["trace_tokens_std"]**2)/2)
        contrasts[name+"_vs_random"]={"relative_mean_length_difference":a["trace_tokens_mean"]/b["trace_tokens_mean"]-1,
                                       "standardized_mean_difference":(a["trace_tokens_mean"]-b["trace_tokens_mean"])/sd}
    deciles=length_deciles([r["n_tokens"] for r in pool])
    bins={r["uid"]:(int(d),r["source"],r["problem_type"]) for r,d in zip(pool,deciles)}
    signatures={name:Counter(bins[r["uid"]] for r in rows) for name,rows in arms.items()}
    samebins=signatures["high"]==signatures["low"]==signatures["random"]
    report={"protocol":"docs/tmlr_matching_feasibility_protocol.md","pool_rows":len(pool),"unique_normalized_questions":len(groups),
            "question_multiplicity":dict(counts),"questions_with_distinct_archived_traces":len(pairs),
            "same_question_pilot_minimum":960,"archived_question_pilot_feasible":len(pairs)>=960,
            "original_bin_counts_identical":samebins,"arms":stats,"length_contrasts":contrasts,
            "length_only_new_bank_priority":"LOW_ALREADY_MATCHED" if samebins and all(abs(c["relative_mean_length_difference"])<=.01 and abs(c["standardized_mean_difference"])<=.1 for c in contrasts.values()) else "REVIEW_RESIDUAL_IMBALANCE",
            "scope":"Qwen bare trace lengths, not canonical supervised lengths or actual sampled exposure; alternate cached source generations not inspected",
            "sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/"data/stage1/pool.jsonl",*[ROOT/f"data/stage1/train_armA_{name}.jsonl" for name in arms]]}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise ValueError("feasibility report exists; preserve it")
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("pool_rows","unique_normalized_questions","questions_with_distinct_archived_traces","original_bin_counts_identical","length_only_new_bank_priority")}))


if __name__=="__main__":main()
