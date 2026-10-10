"""Independent completed-artifact check; writes only a new verification report."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import audit_eos_runtime_readonly as runtime
from ila.config import load_config
from ila.textio import read_jsonl,extract_boxed,answers_match,answers_match_choice
from ila.cost_boundary import measures
from diagnose_stage1_outputs import loop_metrics
from evaluate import file_hash


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise ValueError("verification report exists; preserve it")
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):runtime.main()
    report=json.loads(stream.getvalue())
    issues=report["issues"]
    if report["done_jobs"]!=36 or report["failed_jobs"] or len(report["evaluations_checked"])!=72:
        issues.append({"kind":"incomplete_matrix"})
    pre=json.loads((runtime.CONTROL/"precheck.json").read_text())
    report["model_files_verified"]=0
    for model in pre["model_fingerprints"].values():
        for name,digest in model["sha256"].items():
            if file_hash(Path(model["snapshot"])/name)!=digest:issues.append({"kind":"model_bytes_changed","file":name})
            report["model_files_verified"]+=1
    report["raw_metric_checks"]=[]
    for job in pre["jobs"]:
        cfg=load_config(job["train_config"])
        rec=json.loads((ROOT/cfg["results_dir"]/job["arm"]/f"seed{job['seed']}/run_record.json").read_text())
        for name,digest in rec["code_sha256"].items():
            if file_hash(name)!=digest:issues.append({"kind":"native_helper_drift","job":job["id"],"file":name})
        if rec["status"]!="completed" or rec["global_steps"]!=60 or rec["n_truncated_examples"]:
            issues.append({"kind":"training_dose_failure","job":job["id"]})
        ecfg=load_config(job["eval_config"])
        style=f"lr1e5_step60__{job['arm']}"
        for es in ("svamp","aqua"):
            path=ROOT/ecfg["results_dir"]/style/f"seed{job['seed']}/eval"/es
            rows=read_jsonl(path/"outputs.jsonl");summary=json.loads((path/"summary.json").read_text())
            for r in rows:
                pred=extract_boxed(r["output"])
                correct=answers_match_choice(pred,r["gold"]) if es=="aqua" else answers_match(pred,r["gold"])
                if pred!=r["pred"] or correct!=r["correct"] or len(r["generated_token_ids"])!=r["n_gen_tokens"]:
                    issues.append({"kind":"raw_scoring_mismatch","job":job["id"],"uid":r["uid"]})
            point=measures([r["n_gen_tokens"] for r in rows],[r["truncated"] for r in rows],[r["correct"] for r in rows])
            for key in ("accuracy","mean_tokens","truncation_rate","tokens_per_correct"):
                a,b=point[key],summary[key]
                if (a is None)!=(b is None) or (a is not None and not np.isclose(a,b,rtol=0,atol=1e-10)):
                    issues.append({"kind":"summary_mismatch","job":job["id"],"metric":key})
            point["likely_loop_rate"]=float(np.mean([loop_metrics(r["output"])["likely_loop"] for r in rows]))
            bank=json.loads((ROOT/ecfg["results_dir"]/"analysis.json").read_text())["recipes"]["lr1e5_step60"][es]["arms"][job["arm"]]
            recorded=bank["per_seed"][job["seed"]]
            for key in ("accuracy","mean_tokens","truncation_rate","likely_loop_rate"):
                if not np.isclose(point[key],recorded[key],rtol=0,atol=1e-10):issues.append({"kind":"seed_analysis_mismatch","job":job["id"],"metric":key})
            report["raw_metric_checks"].append({"job":job["id"],"dataset":es,"point":point,"outputs_sha256":file_hash(path/"outputs.jsonl")})
    report["verification_status"]="PASS" if not issues else "FAIL_REVIEW_REQUIRED"
    report["scope"]="all36 completed trainings/all72 evaluations; independently rederived raw scores/cost/loops, hashes and seed summaries; not independent human or publication-final review"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":report["verification_status"],"issues":issues,"evaluations":len(report["raw_metric_checks"])}))


if __name__=="__main__":main()
