"""Atomic queue worker with an enforced physical GPU allowlist."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from ila.config import load_config
from ila.gpu_lease import gpu_lease
ALLOWED={0,1,2,3,4,7}


def ready(gpu):
    raw=subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-gpu=memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip()
    free,util=[int(x.strip()) for x in raw.split(",")]
    return free>=38000 and util<=10


def register(cfg,style,seed,adapter,source=None,step=None):
    d=ROOT/cfg["results_dir"]/style/f"seed{seed}"; d.mkdir(parents=True,exist_ok=True)
    record={"status":"completed","kind":"frozen_adapter_evaluation","model_dir":cfg["model"],
            "adapter_dir":adapter,"max_lora_rank":64,"seed":seed,"config":cfg,
            "source_run_record":source,"optimizer_step":step}
    target=d/"run_record.json"
    if target.exists() and json.loads(target.read_text())!=record:
        raise RuntimeError("conflicting evaluation record; original preserved")
    target.write_text(json.dumps(record,indent=2)+"\n")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--jobs",type=Path,required=True);ap.add_argument("--queue",type=Path,required=True);ap.add_argument("--gpu",type=int,required=True);args=ap.parse_args()
    if args.gpu not in ALLOWED:raise SystemExit("GPU prohibited; allowlist is 0,1,2,3,4,7")
    os.chdir(ROOT)
    jobs=json.loads(args.jobs.read_text()); args.queue.mkdir(parents=True,exist_ok=True)
    for part in ("claims","done","failed","logs"): (args.queue/part).mkdir(exist_ok=True)
    env={**os.environ,"CUDA_VISIBLE_DEVICES":str(args.gpu),"HF_HUB_OFFLINE":"1","PYTHONPATH":"src",
         "ILA_VLLM_UTIL":"0.70","ILA_GPU_MEM_GB":"34","PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}
    train_py=str(ROOT/".venv/bin/python");eval_py=str(ROOT/".venv-vllm/bin/python")
    commands=[]
    def execute(cmd,log):
        commands.append({"command":cmd,"log":str(log)})
        with open(log,"w") as stream:
            subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    def evaluate(cfg_path,style,seed,base,job_id,eval_sets=("svamp","aqua")):
        if not eval_sets or any(es not in ("svamp","aqua") for es in eval_sets):
            raise ValueError("unsupported or empty evaluation set list")
        for es in eval_sets:
            command=[eval_py,"scripts/evaluate.py","--config",cfg_path,"--style",style,"--seed",str(seed),"--eval-set",es]
            if base:command.append("--base")
            execute(command,args.queue/"logs"/f"{job_id}_{style}_{es}.log")
    while True:
        remaining=[j for j in jobs if not (args.queue/"claims"/j["id"]).exists()]
        if not remaining:return
        if not ready(args.gpu):
            print(f"WAIT_GPU gpu={args.gpu} unclaimed={len(remaining)}",flush=True);time.sleep(30);continue
        lease=gpu_lease(ROOT/"results/tmlr_strengthening_controller/gpu_leases",args.gpu)
        owned=lease.__enter__()
        if not owned:
            lease.__exit__(None,None,None);time.sleep(15);continue
        if not ready(args.gpu):
            lease.__exit__(None,None,None);time.sleep(15);continue
        job=remaining[0]
        try:(args.queue/"claims"/job["id"]).mkdir()
        except FileExistsError:
            lease.__exit__(None,None,None);continue
        start=time.time()
        commands=[]
        try:
            if job["mode"]=="eval":
                cfg=load_config(job["config"])
                if not job.get("base"):
                    register(cfg,job["style"],job["seed"],job["adapter"],job["source"],job["step"])
                evaluate(job["config"],job["style"],job["seed"],job.get("base",False),job["id"],job.get("eval_sets",("svamp","aqua")))
            else:
                execute([train_py,"scripts/train_sft.py","--config",job["config"],"--style",job["arm"],"--seed",str(job["seed"])],args.queue/"logs"/f"{job['id']}_train.log")
                train_cfg=load_config(job["config"])
                source=Path(train_cfg["results_dir"])/job["arm"]/f"seed{job['seed']}"
                src_record=json.loads((source/"run_record.json").read_text())
                if src_record["global_steps"]!=60 or not src_record["device"].startswith("cuda") or src_record["n_truncated_examples"]:
                    raise RuntimeError("training dose/device/truncation check failed")
                cfg=load_config("configs/tmlr_external_mistral_resolved.yaml")
                steps=(15,60) if job["lr"]=="1e5" else (60,)
                for step in steps:
                    style=f"lr{job['lr']}_step{step}__{job['arm']}"
                    adapter=str(source/"trajectory_adapters/step015") if step==15 else src_record["adapter_dir"]
                    register(cfg,style,job["seed"],adapter,str(source/"run_record.json"),step)
                    evaluate("configs/tmlr_external_mistral_resolved.yaml",style,job["seed"],False,job["id"])
            result={"job":job,"physical_gpu":args.gpu,"status":"completed","elapsed_seconds":time.time()-start,
                    "commands":commands,"resolved_config":load_config(job["config"]),
                    "environment":{k:env[k] for k in ("CUDA_VISIBLE_DEVICES","HF_HUB_OFFLINE","ILA_VLLM_UTIL","ILA_GPU_MEM_GB")},
                    "gpu_identity":subprocess.check_output(["nvidia-smi",f"--id={args.gpu}","--query-gpu=name,uuid","--format=csv,noheader"],text=True).strip()}
            (args.queue/"done"/f"{job['id']}.json").write_text(json.dumps(result,indent=2))
            print(f"DONE gpu={args.gpu} id={job['id']}",flush=True)
        except Exception as error:
            (args.queue/"failed"/f"{job['id']}.json").write_text(json.dumps({"job":job,"gpu":args.gpu,"error":repr(error),"commands":commands},indent=2))
            print(f"FAILED gpu={args.gpu} id={job['id']} error={error!r}",flush=True)
        finally:
            lease.__exit__(None,None,None)


if __name__=="__main__":main()
