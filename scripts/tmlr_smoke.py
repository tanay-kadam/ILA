"""Device-verified smoke on the same enforced allowlist as full workers."""
import argparse
import os
import subprocess
import time
from pathlib import Path
from tmlr_gpu_worker import ALLOWED,ready
from ila.gpu_lease import gpu_lease


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--family",choices=("qwen","mistral"),required=True);args=ap.parse_args()
    while True:
        available=[g for g in sorted(ALLOWED) if ready(g)]
        lease=None
        for candidate in available:
            candidate_lease=gpu_lease(Path("results/tmlr_strengthening_controller/gpu_leases"),candidate)
            if candidate_lease.__enter__():
                lease=candidate_lease;gpu=candidate;break
            candidate_lease.__exit__(None,None,None)
        if lease is not None:break
        print("WAIT_SMOKE_CAPACITY allowlist=0,1,2,3,4,7",flush=True);time.sleep(30)
    env={**os.environ,"CUDA_VISIBLE_DEVICES":str(gpu),"HF_HUB_OFFLINE":"1","PYTHONPATH":"src","ILA_VLLM_UTIL":"0.70","ILA_GPU_MEM_GB":"34"}
    print(f"SMOKE_DEVICE physical_gpu={gpu} family={args.family}",flush=True)
    import atexit
    atexit.register(lambda: lease.__exit__(None,None,None))
    config=f"configs/tmlr_external_{args.family}"+("_resolved.yaml" if args.family=="mistral" else ".yaml")
    style="_base"
    if args.family=="mistral":
        subprocess.run([".venv/bin/python","scripts/train_sft.py","--config","configs/tmlr_mistral_lr1e5_resolved.yaml","--style","armA_random","--seed","999","--set","results_dir=results/tmlr_mistral_smoke","train.stop_after_steps=1","train.snapshot_steps=[]"],env=env,check=True)
        from ila.config import load_config
        from tmlr_gpu_worker import register
        import json
        cfg=load_config(config);cfg["results_dir"]="results/tmlr_eval_smoke_mistral"
        rec=json.loads(Path("results/tmlr_mistral_smoke/armA_random/seed999/run_record.json").read_text())
        if rec["global_steps"]!=1 or rec["n_truncated_examples"] or not rec["device"].startswith("cuda"):
            raise RuntimeError("smoke train integrity failed")
        register(cfg,"adapter_smoke",999,rec["adapter_dir"])
        style="adapter_smoke"
    for es in ("svamp","aqua"):
        cmd=[".venv-vllm/bin/python","scripts/evaluate.py","--config",config,"--style",style,"--seed","999" if args.family=="mistral" else "0","--eval-set",es,"--limit","2","--set",f"results_dir=results/tmlr_eval_smoke_{args.family}"]
        if args.family=="qwen":cmd.append("--base")
        subprocess.run(cmd,env=env,check=True)
    print(f"SMOKE_PASSED family={args.family} physical_gpu={gpu}",flush=True)


if __name__=="__main__":main()
