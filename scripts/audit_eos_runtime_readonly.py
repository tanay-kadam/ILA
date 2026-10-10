"""Read-only EOS audit; prints evidence, never modifies runs or artifacts."""
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from ila.config import load_config
from ila.textio import read_jsonl
from evaluate import evaluation_fingerprint,file_hash
CONTROL=ROOT/"results/tmlr_eos_control_v3/controller"


def main():
    pre=json.loads((CONTROL/"precheck.json").read_text())
    jobs=json.loads((CONTROL/"jobs.json").read_text())
    groups={}
    issues=[]
    for job in jobs:
        groups.setdefault((job["family"],job["arm"],job["seed"]),{})[job["weight"]]=job
    for key,pair in groups.items():
        if set(pair)!={1,10}:raise ValueError("missing paired weight")
        for kind in ("train_config","eval_config"):
            a,b=[copy.deepcopy(load_config(pair[w][kind])) for w in (1,10)]
            for cfg in (a,b):
                cfg.pop("results_dir")
                cfg["train"].pop("terminal_eos_weight")
            if a!=b:issues.append({"kind":"pair_config_drift","pair":key,"config":kind})
        if pair[1]["train_sha256"]!=pair[10]["train_sha256"]:issues.append({"kind":"pair_data_drift","pair":key})
    hashes={name:{"expected":digest,"actual":file_hash(ROOT/name)} for name,digest in pre["code_sha256"].items()}
    scheduler=json.loads((CONTROL/"scheduler_fix.json").read_text())
    for name,values in hashes.items():
        if values["expected"]!=values["actual"]:
            if name!="scripts/tmlr_eos_control.py" or values["actual"]!=scheduler["worker_controller_sha256"]:
                issues.append({"kind":"unexplained_code_drift","file":name})
    records=[];evaluations=[]
    for job in jobs:
        cfg=load_config(job["train_config"])
        path=ROOT/cfg["results_dir"]/job["arm"]/f"seed{job['seed']}"
        recpath=path/"run_record.json"
        if not recpath.exists():continue
        rec=json.loads(recpath.read_text())
        if rec["config"]!=cfg:issues.append({"kind":"runtime_config_drift","job":job["id"]})
        if rec["train_file_sha256"]!=job["train_sha256"]:issues.append({"kind":"runtime_data_drift","job":job["id"]})
        if int(rec["cuda_visible_devices"]) not in pre["allowlist"]:issues.append({"kind":"forbidden_gpu","job":job["id"]})
        term=json.loads((path/"terminal_eos_record.json").read_text())
        for name,digest in term["source_sha256"].items():
            if file_hash(name)!=digest:issues.append({"kind":"training_source_drift","job":job["id"],"file":name})
        records.append({"id":job["id"],"status":rec["status"],"weight":term["weight"],"steps":rec.get("global_steps"),
            "loss_calls":term.get("loss_calls"),"dataset_eligible_terminal_examples":term.get("weighted_terminal_examples"),
            "nominal_examples_at60":60*cfg["train"]["per_device_batch_size"]*cfg["train"]["grad_accum"],
            "model_accepts_loss_kwargs":term.get("model_accepts_loss_kwargs"),"accumulated_denominator":term.get("received_accumulated_denominator")})
        ecfg=load_config(job["eval_config"])
        style=f"lr1e5_step60__{job['arm']}"
        edir=ROOT/ecfg["results_dir"]/style/f"seed{job['seed']}"
        for es in ("svamp","aqua"):
            summary_path=edir/"eval"/es/"summary.json"
            if not summary_path.exists():continue
            summary=json.loads(summary_path.read_text())
            erec=json.loads((edir/"run_record.json").read_text())
            data=ROOT/ecfg["data_dir"]/ecfg["eval"]["sets"][es]["file"]
            fingerprint=evaluation_fingerprint(ecfg,erec,data,style,job["seed"],None,True)
            if fingerprint!=summary["evaluation_fingerprint"]:issues.append({"kind":"stale_eval_fingerprint","job":job["id"],"dataset":es})
            if not erec.get("adapter_dir"):issues.append({"kind":"missing_eval_adapter","job":job["id"]})
            raw=read_jsonl(summary_path.with_name("outputs.jsonl"));frozen=read_jsonl(data)
            if [(r["uid"],r["gold"]) for r in raw]!=[(r["uid"],r["gold"]) for r in frozen]:issues.append({"kind":"eval_UID_drift","job":job["id"]})
            evaluations.append({"id":job["id"],"dataset":es,"n":summary["n"],"fingerprint_matches":fingerprint==summary["evaluation_fingerprint"]})
    report={"audit":"READ_ONLY_CURRENT_EOS_RUNTIME", "paired_config_groups":len(groups),"issues":issues,
        "runtime_records":records,"evaluations_checked":evaluations,
        "done_jobs":len(list((CONTROL/"done").glob("*.json"))),"failed_jobs":len(list((CONTROL/"failed").glob("*.json"))),
        "current_status":(CONTROL/"STATUS").read_text().strip(),"source_hash_checks":hashes,
        "scope":"current available artifacts only; incomplete runs not certified completed"}
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
