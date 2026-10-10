"""Audit submitted AI labels and prepare unfilled author-adjudication cases."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from ila.textio import read_jsonl,write_jsonl
base=ROOT/"data/human_review_v1"
items={r["item_id"]:r for r in read_jsonl(base/"reviewer_items.jsonl")}
names=("reviewer_A_labels_AI_DRAFT.jsonl","reviewer_B_AI_provisional.jsonl")
maps=[];report={"kind":"PROVISIONAL_AI_REVIEW_NOT_HUMAN_VALIDATION","files":{},"tasks":{}}
for name in names:
    rows=read_jsonl(base/name);mapping={r["item_id"]:r for r in rows}
    if len(rows)!=127 or len(mapping)!=127 or set(mapping)!=set(items):raise ValueError("expected127 unique matching IDs")
    for uid,r in mapping.items():
        task=items[uid]["task"];allowed={"yes","no","uncertain"} if task=="loop_assessment" else set("ABCDE")|{"ambiguous","unparseable"}
        if r["task"]!=task or r["decision"] not in allowed:raise ValueError("invalid task/decision")
    maps.append(mapping)
    report["files"][name]={"sha256":hashlib.sha256((base/name).read_bytes()).hexdigest(),"n":len(rows),
        "declared_reviewer_ids":sorted({r["reviewer_id"] for r in rows}),"counts":dict(Counter(r["task"]+"/"+r["decision"] for r in rows))}
a,b=maps;cases=[]
for task in ("loop_assessment","final_choice_interpretation"):
    ids=[uid for uid,r in items.items() if r["task"]==task]
    conflicts=[uid for uid in ids if a[uid]["decision"]!=b[uid]["decision"]]
    report["tasks"][task]={"n":len(ids),"agreements":len(ids)-len(conflicts),"raw_ai_concordance":(len(ids)-len(conflicts))/len(ids),"disagreement_ids":conflicts}
    for uid in conflicts:
        cases.append({**items[uid],"ai_a":a[uid],"ai_b_provisional":b[uid],"author_decision":None,"author_reason":"",
            "review_status":"PENDING_SINGLE_AUTHOR_AI_ASSISTED_TARGETED_ADJUDICATION_NOT_BLINDED_VALIDATION"})
report["total_disagreements"]=len(cases)
report["actual_human_labels"]=0
report["missing_options_claims"]=[uid for uid,r in b.items() if ("not supplied" in r["reason"] or "not provided" in r["reason"]) and items[uid].get("options")]
report["limitations"]=["self-reported AI provenance, not provider-verified model/version/input logs", "raw concordance is not accuracy or human agreement", "targeted adjudication is unblinded and does not validate the whole sample/population", "provisional rerun follows input/rubric feedback; not pristine confirmatory evidence"]
out=ROOT/"docs/ai_review_provisional_audit.json"
template=base/"author_adjudication_AI_DISAGREEMENTS_PENDING.jsonl"
if out.exists() or template.exists():raise ValueError("existing audit/adjudication files preserved")
out.write_text(json.dumps(report,indent=2)+"\n")
write_jsonl(template,cases)
print(json.dumps({"tasks":report["tasks"],"total_disagreements":len(cases),"actual_human_labels":0}))
