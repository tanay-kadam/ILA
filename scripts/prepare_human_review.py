"""Prepare blinded reviewer items and empty label templates, never human labels."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from ila.textio import read_jsonl,write_jsonl
from diagnose_stage1_outputs import loop_metrics
OUT=ROOT/"data/human_review_v1"


def main():
    if OUT.exists():raise ValueError("review bundle exists; preserved")
    pools=defaultdict(list);counts=Counter();chosen=[]
    case_sources={}
    for family in ("qwen","mistral"):
        path=ROOT/f"docs/tmlr_format_audit_{family}.cases.jsonl"
        if not path.exists():path=ROOT/f"results/tmlr_format_audit_original/{family}.cases.jsonl"
        if not path.exists():raise ValueError("original diagnostic case source missing")
        case_sources[family]=path
    source=ROOT/"results/tmlr_eos_control_v3"
    for family in ("qwen","mistral"):
        for weight in (1,10):
            for path in sorted((source/family/f"weight{weight}/eval").glob("*/seed*/eval/*/outputs.jsonl")):
                es=path.parent.name
                questions={r["uid"]:r["question"] for r in read_jsonl(ROOT/f"data/tmlr_external/eval_{es}.jsonl")}
                for row in read_jsonl(path):
                    flag=bool(loop_metrics(row["output"])["likely_loop"])
                    stratum=(family,weight,flag,bool(row["truncated"]))
                    counts[stratum]+=1
                    key=hashlib.sha256(f"review20261009|{path}|{row['uid']}".encode()).hexdigest()
                    item={"item_id":key[:20],"task":"loop_assessment","question":questions[row["uid"]],"output":row["output"]}
                    private={"item_id":item["item_id"],"source":str(path),"uid":row["uid"],"stratum":stratum,
                             "automatic_loop":flag,"truncated":row["truncated"],"output_sha256":hashlib.sha256(row["output"].encode()).hexdigest()}
                    pools[stratum].append((key,item,private))
                    pools[stratum]=sorted(pools[stratum],key=lambda x:x[0])[:6]
    for values in pools.values():chosen.extend(values)
    choice_seen=set()
    for family in ("qwen","mistral"):
        path=case_sources[family]
        for row in read_jsonl(path):
            identity=(row["question"],row["output"])
            if identity in choice_seen:continue
            choice_seen.add(identity)
            key=hashlib.sha256(f"choice20261009|{family}|{row['run']}|{row['uid']}".encode()).hexdigest()
            item={"item_id":key[:20],"task":"final_choice_interpretation","question":row["question"],"options":row["options"],"output":row["output"]}
            private={"item_id":item["item_id"],"family":family,"source_case_file":str(path),"run":row["run"],"uid":row["uid"],
                     "automated_mapping":row["mapping"],"gold":row["gold"],"sampling":"purposeful ambiguous/non-strict cases; not a population-accuracy sample"}
            chosen.append((key,item,private))
    chosen.sort(key=lambda x:x[0])
    OUT.mkdir(parents=True)
    write_jsonl(OUT/"reviewer_items.jsonl",[item for _,item,_ in chosen])
    write_jsonl(OUT/"private_key_DO_NOT_SHARE_WITH_REVIEWERS.jsonl",[private for _,_,private in chosen])
    for reviewer in ("A","B"):
        write_jsonl(OUT/f"reviewer_{reviewer}_labels_EMPTY.jsonl",[
            {"item_id":item["item_id"],"reviewer_id":"","task":item["task"],"decision":None,"reason":""} for _,item,_ in chosen])
    manifest={"status":"PREPARED_NOT_HUMAN_REVIEWED","n":len(chosen),"tasks":dict(Counter(item["task"] for _,item,_ in chosen)),
        "loop_sampling":"six fixed-SHA lowest items per family/weight/automatic-loop/cap stratum, or all if fewer",
        "loop_stratum_population":{str(k):v for k,v in counts.items()},"loop_stratum_sample":{str(k):len(v) for k,v in pools.items()},
        "blinding":"explicit model/weight/arm/seed/automatic flags/gold hidden; text may reveal clues; do not claim perfect blinding",
        "labels":"EMPTY; AI has not supplied human decisions","source":"completed EOS outputs plus previously exported original AQuA diagnostic cases",
        "caveats":["stratified loop sample needs population weighting for rate estimates", "choice sample is purposeful, not unbiased accuracy validation", "choice interpretation is not independent mathematical reasoning verification"]}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest))


if __name__=="__main__":main()
