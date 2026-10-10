"""Validate actual reviewer files and report descriptive agreement; never invent labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from ila.textio import read_jsonl


def validate_labels(items,path):
    rows=read_jsonl(path)
    if len(rows)!=len(items) or len({r["item_id"] for r in rows})!=len(items):raise ValueError("complete unique item coverage required")
    mapping={r["item_id"]:r for r in rows}
    if set(mapping)!=set(items):raise ValueError("review item IDs differ")
    identities={r.get("reviewer_id","").strip() for r in rows}
    if len(identities)!=1 or not next(iter(identities)):raise ValueError("one genuine declared reviewer identifier per file required")
    for uid,item in items.items():
        r=mapping[uid]
        allowed={"yes","no","uncertain"} if item["task"]=="loop_assessment" else set("ABCDE")|{"ambiguous","unparseable"}
        if r.get("task")!=item["task"] or r.get("decision") not in allowed:raise ValueError("missing/invalid human decision; empty templates are not labels")
    return next(iter(identities)),mapping


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--labels-a",type=Path,required=True);ap.add_argument("--labels-b",type=Path)
    ap.add_argument("--attest-human-review",action="store_true");ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    if not args.attest_human_review:raise ValueError("author must attest actual human review; AI labels cannot satisfy this task")
    if args.output.exists():raise ValueError("prior report preserved")
    root=ROOT/"data/human_review_v1"
    items={r["item_id"]:r for r in read_jsonl(root/"reviewer_items.jsonl")}
    name_a,a=validate_labels(items,args.labels_a)
    report={"status":"DECLARED_HUMAN_LABELS_VALIDATED_NOT_IDENTITY_VERIFIED", "reviewer_a":name_a,"n":len(items),
            "caveat":"provenance attested by author, not independently proven by software; purposeful choice sample is not benchmark accuracy validation",
            "reviewer_counts":{name_a:dict(Counter(f"{r['task']}/{r['decision']}" for r in a.values()))}}
    if args.labels_b:
        name_b,b=validate_labels(items,args.labels_b)
        if name_a==name_b:raise ValueError("same reviewer cannot count as two independent label sets")
        report["reviewer_b"]=name_b
        report["reviewer_counts"][name_b]=dict(Counter(f"{r['task']}/{r['decision']}" for r in b.values()))
        report["agreement"]={task:{"n":sum(x["task"]==task for x in items.values()),
            "exact_agreement":sum(a[uid]["decision"]==b[uid]["decision"] for uid,x in items.items() if x["task"]==task)/sum(x["task"]==task for x in items.values())}
            for task in {x["task"] for x in items.values()}}
        report["disagreement_item_ids"]=[uid for uid in items if a[uid]["decision"]!=b[uid]["decision"]]
    else:report["agreement_status"]="NOT_AVAILABLE_SINGLE_REVIEWER"
    report["loop_label_counts_sample_only"]={decision:sum(r["task"]=="loop_assessment" and r["decision"]==decision for r in a.values()) for decision in ("yes","no","uncertain")}
    report["analysis_limits"]=["raw sample counts/agreement only; no population loop-error rate inferred without stratification weighting/uncertainty", "no human mathematical correctness scores inferred from option interpretation", "no AI/human equivalence or perfect blinding claim"]
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"n":report["n"]}))


if __name__=="__main__":main()
