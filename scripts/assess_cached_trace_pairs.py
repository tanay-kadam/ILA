"""Outcome-blind same-question feasibility using only cached original source shards."""
import argparse
from collections import Counter,defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from ila.textio import read_jsonl,build_chat_user_content
from ila.natural import words,ngram_set,is_contaminated
from ila.features import count_reflections
from evaluate import file_hash


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise ValueError("report exists; preserved")
    from datasets import Dataset
    from transformers import AutoTokenizer
    cache=Path("/home/tkadam/.cache/huggingface/datasets/open-r1___open_r1-math-220k/default/0.0.0/e4e141ec9dea9f8326f4d347be56105859b2bd68")
    shards=sorted(cache.glob("*-train-*.arrow"))
    if len(shards)!=10:raise ValueError("original10 cached train shards missing")
    pool=read_jsonl(ROOT/"data/stage1/pool.jsonl")
    # Preserve operators, variable case and LaTeX; fold whitespace only.
    allowed={" ".join(r["question"].split()) for r in pool}
    references=[r["question"] for es in ("svamp","aqua") for r in read_jsonl(ROOT/f"data/tmlr_external/eval_{es}.jsonl")]
    grams,short=ngram_set(references,13)
    exact={tuple(words(q)) for q in references}
    reject=Counter();groups=defaultdict(dict);counts=Counter()
    source_hashes={str(p):file_hash(p) for p in shards}
    for shard in shards:
        ds=Dataset.from_file(str(shard))
        for index,row in enumerate(ds):
            counts["source_rows"]+=1
            q=row["problem"].strip();key=" ".join(q.split())
            if key not in allowed:continue
            counts["rows_on_original_pool_questions"]+=1
            if tuple(words(q)) in exact or is_contaminated(q,grams,short,13):reject["external_overlap_question"]+=1;continue
            if row.get("finish_reasons") is None:reject["missing_stop_provenance"]+=1;continue
            for i,target in enumerate(row.get("generations") or []):
                correct=row.get("correctness_math_verify") or []
                complete=row.get("is_reasoning_complete") or []
                reasons=row["finish_reasons"]
                if i>=len(correct) or not correct[i] or i>=len(complete) or not complete[i] or i>=len(reasons) or reasons[i]!="stop":continue
                target=(target or "").strip()
                if not 1500<=len(target)<=24000:continue
                digest=hashlib.sha256(target.encode()).hexdigest()
                groups[key][digest]={"question":q,"target":target,"sha256":digest,"source_shard":str(shard),"source_row":index,"generation":i}
    counts["question_groups_two_verified_traces"]=sum(len(v)>=2 for v in groups.values())
    pre=json.loads((ROOT/"results/tmlr_eos_control_v3/controller/precheck.json").read_text())
    tokenizers={f:AutoTokenizer.from_pretrained(d["snapshot"]) for f,d in pre["model_fingerprints"].items()}
    configs={f:next(s["resolved_config"] for s in pre["sources"] if s["pair"].startswith(f+"_")) for f in tokenizers}
    candidates=[]
    for key,members in groups.items():
        if len(members)<2:continue
        traces=[]
        for item in members.values():
            bare=len(tokenizers["qwen"](item["target"],add_special_tokens=False)["input_ids"])
            if not 1000<=bare<=2500:continue
            lengths={};fits=True
            for family,tok in tokenizers.items():
                cfg=configs[family]
                user={"role":"user","content":build_chat_user_content(item["question"],cfg["prompt_template"])}
                prefix=tok.apply_chat_template([user],tokenize=True,add_generation_prompt=True)
                full=tok.apply_chat_template([user,{"role":"assistant","content":item["target"]}],tokenize=True,add_generation_prompt=False)
                if hasattr(prefix,"keys"):prefix=prefix["input_ids"]
                if hasattr(full,"keys"):full=full["input_ids"]
                if full[:len(prefix)]!=prefix:raise ValueError("canonical prefix mismatch")
                if len(full)>cfg["train"]["max_len"]:fits=False;break
                lengths[family]=len(full)-len(prefix)
            if fits:traces.append({**item,"bare_tokens":bare,"density":1000*count_reflections(item["target"])/bare,"lengths":lengths})
        eligible=[]
        for a,b in itertools.combinations(traces,2):
            gaps={f:abs(a["lengths"][f]-b["lengths"][f])/((a["lengths"][f]+b["lengths"][f])/2) for f in tokenizers}
            separation=abs(a["density"]-b["density"])
            if max(gaps.values())<=.01 and separation>=2:
                high,low=(a,b) if a["density"]>b["density"] else (b,a)
                eligible.append((max(gaps.values()),-separation,high["sha256"],low["sha256"],high,low,gaps))
        if eligible:
            _,_,_,_,high,low,gaps=min(eligible,key=lambda x:x[:4])
            fields=("sha256","source_shard","source_row","generation","bare_tokens","density","lengths")
            candidates.append({"question_sha256":hashlib.sha256(key.encode()).hexdigest(),
                "high":{k:high[k] for k in fields},"low":{k:low[k] for k in fields},"relative_length_gaps":gaps})
    report={"protocol":"docs/tmlr_matching_feasibility_protocol.md","source_counts":dict(counts),"rejections":dict(reject),
        "eligible_distinct_questions":len(candidates),"required_distinct_questions":960,"feasible":len(candidates)>=960,
        "margins":{"reflection_density_difference_min":2,"canonical_length_relative_difference_max":.01,"qwen_bare_length_range":[1000,2500]},
        "selection":"source validity/archived question membership/length/reflection only; no model evaluation outcomes; fixed margins not relaxed",
        "source_sha256":source_hashes,"pool_sha256":file_hash(ROOT/"data/stage1/pool.jsonl"),"candidates":candidates,
        "question_identity":"exact text after whitespace folding; operators, variable case and LaTeX preserved",
        "code_sha256":file_hash(Path(__file__)),
        "caveats":["whitespace normalization can merge formatting variants", "verified-correct filtered source subset", "reflection/content not isolated causal interventions"]}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("source_counts","eligible_distinct_questions","required_distinct_questions","feasible")}))


if __name__=="__main__":main()
