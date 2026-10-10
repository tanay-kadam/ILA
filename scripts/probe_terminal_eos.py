"""Read-only teacher-forced EOS diagnostics on every frozen base/final adapter."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import math
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from ila.config import load_config
from ila.textio import read_jsonl
from ila.gpu_lease import gpu_lease
from tmlr_gpu_worker import ALLOWED,ready
from evaluate import file_hash
OUT=ROOT/"results/tmlr_eos_effectiveness_probe"


def run(family):
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/f"{family}.json"
    if target.exists():raise ValueError("probe report exists; preserved")
    lease=None
    while lease is None:
        for gpu in sorted(ALLOWED):
            if not ready(gpu):continue
            candidate=gpu_lease(ROOT/"results/tmlr_strengthening_controller/gpu_leases",gpu)
            if candidate.__enter__():
                if ready(gpu):lease=candidate;break
                candidate.__exit__(None,None,None)
            else:candidate.__exit__(None,None,None)
        if lease is None:print("WAIT_ALLOWED_GPU_CAPACITY",flush=True);time.sleep(30)
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE="1")
        import torch,numpy as np
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from peft import PeftModel
        from train_sft import SFTDataset
        from ila.terminal_eos_loss import terminal_eos_extra,validate_terminal_dataset
        if not torch.cuda.is_available():raise ValueError("no GPU; no CPU model fallback")
        torch.cuda.set_per_process_memory_fraction(min(1,34*2**30/torch.cuda.get_device_properties(0).total_memory))
        source=ROOT/"results/tmlr_eos_control_v3/controller"
        jobs=[j for j in json.loads((source/"jobs.json").read_text()) if j["family"]==family]
        if len(jobs)!=18:raise ValueError("frozen adapter matrix incomplete")
        cfg=load_config(jobs[0]["train_config"]);tok=AutoTokenizer.from_pretrained(cfg["model"])
        banks={};suffix_ids={};sample_info={}
        for arm in ("armA_high","armA_low","armA_random"):
            rows=read_jsonl(ROOT/cfg["data_dir"]/f"train_{arm}.jsonl")
            rows=sorted(rows,key=lambda r:hashlib.sha256(f"probe20261009|{r['uid']}|{r['question']}|{r['target']}".encode()).hexdigest())[:32]
            ds=SFTDataset(rows,tok,cfg["train"]["max_len"],cfg["prompt_template"],canonical_chat=True)
            suffix=set();validate_terminal_dataset(ds,tok.eos_token_id,tok,suffix)
            banks[arm]=ds;suffix_ids[arm]=suffix
            sample_info[arm]=[{"uid":r["uid"],"target_sha256":hashlib.sha256(r["target"].encode()).hexdigest()} for r in rows]
        model=AutoModelForCausalLM.from_pretrained(cfg["model"],dtype=torch.bfloat16).to("cuda").eval()
        report={"family":family,"device":torch.cuda.get_device_name(),"physical_gpu":gpu,"model":cfg["model"],
            "protocol_sha256":file_hash(ROOT/"docs/tmlr_eos_effectiveness_probe_protocol.md"),"code_sha256":file_hash(Path(__file__)),
            "samples":sample_info,"conditions":{},"scope":"posthoc teacher-forced eval-mode scalar-loss diagnostic; not training gradient history or free-running termination"}
        def measure(item,arm):
            ids=torch.tensor([item["input_ids"]],device="cuda")
            labels=torch.tensor([item["labels"]],device="cuda")
            n=int(labels[:,1:].ne(-100).sum().item())
            with torch.inference_mode():
                result=model(input_ids=ids,labels=labels,attention_mask=torch.ones_like(ids),use_cache=False)
                native=float(result.loss.item())
                extra=float(terminal_eos_extra(result.logits,labels,tok.eos_token_id,allowed_suffix_ids=suffix_ids[arm]).item())
            if not math.isfinite(native) or not math.isfinite(extra) or native<0 or extra<0:raise ValueError("nonfinite/negative CE")
            return {"supervised_tokens":n,"native_token_ce_sum":native*n,"terminal_eos_ce":extra*n,
                    "terminal_eos_probability":math.exp(-extra*n),"added_ce_share":9*extra/native if native>0 else None}
        smoke=[measure(item,"armA_random") for item in banks["armA_random"].items[:2]]
        report["smoke"]={"status":"PASS","n":2,"rows":smoke,"gpu":gpu}
        print(f"SMOKE_PASS {family} gpu={gpu}",flush=True)
        def condition(name,arm,metadata):
            rows=[measure(item,arm) for item in banks[arm].items]
            numerator=9*sum(r["terminal_eos_ce"] for r in rows);denominator=sum(r["native_token_ce_sum"] for r in rows)
            share=numerator/denominator if denominator>0 else None
            report["conditions"][name]={"arm":arm,**metadata,"n":len(rows),"added_objective_share":share,
                "small_share_screen_lt_1percent":share is not None and share<.01,
                "terminal_eos_ce_quantiles":np.quantile([r["terminal_eos_ce"] for r in rows],[0,.5,.95,1]).tolist(),"rows":rows}
            print(f"DONE {family} {name} R={share}",flush=True)
            torch.cuda.empty_cache()
        for arm in banks:condition("base__"+arm,arm,{"kind":"base"})
        for index,job in enumerate(jobs):
            tcfg=load_config(job["train_config"])
            record_path=ROOT/tcfg["results_dir"]/job["arm"]/f"seed{job['seed']}/run_record.json"
            rec=json.loads(record_path.read_text())
            if rec["status"]!="completed" or rec["global_steps"]!=60 or not rec.get("adapter_dir"):raise ValueError("invalid frozen adapter record")
            name=f"cp{index}"
            if index==0:model=PeftModel.from_pretrained(model,rec["adapter_dir"],adapter_name=name,is_trainable=False)
            else:model.load_adapter(rec["adapter_dir"],adapter_name=name,is_trainable=False,torch_device="cuda")
            model.set_adapter(name,inference_mode=True);model.eval()
            for parameter in model.parameters():parameter.requires_grad_(False)
            condition(job["id"],job["arm"],{"kind":"adapter","weight":job["weight"],"seed":job["seed"],
                "adapter":rec["adapter_dir"],"source_record_sha256":file_hash(record_path)})
        report["status"]="COMPLETED_VERIFIED_FROZEN_CHECKPOINT_DIAGNOSTIC"
        report["peak_gpu_allocated_gib"]=torch.cuda.max_memory_allocated()/2**30
        target.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    finally:lease.__exit__(None,None,None)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--family",choices=("qwen","mistral"),required=True);args=ap.parse_args()
    os.chdir(ROOT)
    try:run(args.family)
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/f"{args.family}_failure.json").write_text(json.dumps({"status":"FAILED_PARTIAL_DIAGNOSTIC","error":repr(error)},indent=2));raise


if __name__=="__main__":main()
