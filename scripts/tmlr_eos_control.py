"""Detached EOS control: frozen precheck, GPU smokes, 36 matched train/eval jobs."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from ila.config import load_config
from ila.gpu_lease import gpu_lease
from ila.textio import read_jsonl
from evaluate import file_hash
from tmlr_gpu_worker import ALLOWED, ready, register

RUNNAME="tmlr_eos_control_v3"
OUT = ROOT/"results"/RUNNAME
CONTROL = OUT/"controller"
PY = str(ROOT/".venv/bin/python")
EPY = str(ROOT/".venv-vllm/bin/python")
LEASES = ROOT/"results/tmlr_strengthening_controller/gpu_leases"


def status(value):
    (CONTROL/"STATUS").write_text(value+" "+time.strftime("%Y-%m-%dT%H:%M:%S")+"\n")
    print(value, flush=True)


def acquire(gpu=None):
    while True:
        if gpu is not None:
            jobs=json.loads((CONTROL/"jobs.json").read_text())
            if all((CONTROL/"claims"/j["id"]).exists() for j in jobs):
                return None,None
        for device in ([gpu] if gpu is not None else sorted(ALLOWED)):
            if device not in ALLOWED:
                raise ValueError("forbidden GPU")
            if not ready(device):
                continue
            lease = gpu_lease(LEASES, device)
            if lease.__enter__():
                if ready(device):
                    return device, lease
                lease.__exit__(None,None,None)
            else:
                lease.__exit__(None,None,None)
        print(f"WAIT_CAPACITY gpu={gpu} allowlist=0,1,2,3,4,7", flush=True)
        time.sleep(30)


def prepare():
    if OUT.exists():
        raise ValueError("EOS namespace exists; inspect before a versioned retry")
    CONTROL.mkdir(parents=True)
    status("PRECHECK")
    if shutil.disk_usage(ROOT).free/2**30 < 90:
        raise ValueError("less than 90 GiB free; preserve existing evidence and review storage before training")
    import torch, transformers
    from train_sft import SFTDataset
    from ila.terminal_eos_loss import validate_terminal_dataset
    versions = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__}
    inventory = {"versions": versions, "allowlist": sorted(ALLOWED), "jobs": [],
                 "protocol_sha256": file_hash(ROOT/"docs/tmlr_eos_control_protocol.md"),
                 "manifest_sha256": file_hash(ROOT/"data/tmlr_external/manifest.json"), "sources": [],"model_fingerprints":{}}
    checked = {}
    for family in ("qwen","mistral"):
        for arm in ("armA_high","armA_low","armA_random"):
            for seed in range(3):
                evaluation_path = ROOT/f"results/tmlr_external_{family}/lr1e5_step60__{arm}/seed{seed}/run_record.json"
                evaluation = json.loads(evaluation_path.read_text())
                if evaluation["optimizer_step"] != 60:
                    raise ValueError("wrong comparator step")
                adapter = ROOT/evaluation["adapter_dir"]
                source_path = None
                # Trace evaluation-only aliases to the real adapter-producing run.
                for parent in adapter.parents:
                    path = parent/"run_record.json"
                    if path.exists():
                        possible = json.loads(path.read_text())
                        if possible.get("versions") and possible.get("config") and possible.get("n_train"):
                            source_path, source = path, possible
                            break
                    if parent == ROOT:
                        break
                if source_path is None:
                    raise ValueError(f"missing original training provenance: {adapter}")
                if source["status"] != "completed" or source["seed"] != seed or source["n_truncated_examples"]:
                    raise ValueError("invalid source training state")
                cfg = copy.deepcopy(source["config"])
                if family not in inventory["model_fingerprints"]:
                    from huggingface_hub import snapshot_download
                    model_path=Path(cfg["model"])
                    if not model_path.is_dir():
                        model_path=Path(snapshot_download(cfg["model"],local_files_only=True))
                    files=[p for p in model_path.iterdir() if p.is_file() and
                           (p.suffix==".safetensors" or p.name.endswith(".json") or "tokenizer" in p.name or p.name.endswith(".jinja"))]
                    if not any(p.suffix==".safetensors" for p in files):raise ValueError("no pinned model weight files")
                    inventory["model_fingerprints"][family]={"snapshot":str(model_path),"sha256":{p.name:file_hash(p) for p in files}}
                cfg["model"]=inventory["model_fingerprints"][family]["snapshot"]
                if cfg["train"]["learning_rate"] != 1e-5 or cfg["train"]["max_steps"] != 230:
                    raise ValueError("source recipe does not match frozen LR/horizon")
                train_file = ROOT/cfg["data_dir"]/f"train_{arm}.jsonl"
                if source.get("train_file_sha256") and file_hash(train_file) != source["train_file_sha256"]:
                    raise ValueError("source training data changed")
                cfg["experiment"] = f"tmlr_eos10_{family}_{arm}_seed{seed}"
                cfg["results_dir"] = f"results/{RUNNAME}/{family}/train"
                cfg["train"].update(stop_after_steps=60,save_merged_model=False)
                key = (cfg["model"], str(train_file), cfg["train"]["max_len"], cfg["prompt_template"])
                if key not in checked:
                    tok = transformers.AutoTokenizer.from_pretrained(cfg["model"])
                    ds = SFTDataset(read_jsonl(train_file), tok, cfg["train"]["max_len"], cfg["prompt_template"], canonical_chat=True)
                    checked[key] = validate_terminal_dataset(ds,tok.eos_token_id,tok)
                ecfg = copy.deepcopy(evaluation["config"])
                ecfg["model"]=cfg["model"]
                ecfg["experiment"] = cfg["experiment"]+"_eval"
                ecfg["results_dir"] = f"results/{RUNNAME}/{family}/eval"
                for weight in (1,10):
                    cfg["train"]["terminal_eos_weight"]=float(weight)
                    cfg["results_dir"]=f"results/{RUNNAME}/{family}/weight{weight}/train"
                    cfg_path=CONTROL/f"{family}_weight{weight}_{arm}_seed{seed}_train.yaml"
                    cfg_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
                    ecfg["results_dir"]=f"results/{RUNNAME}/{family}/weight{weight}/eval"
                    ecfg["train"]=copy.deepcopy(cfg["train"])
                    eval_path=CONTROL/f"{family}_weight{weight}_{arm}_seed{seed}_eval.yaml"
                    eval_path.write_text(yaml.safe_dump(ecfg,sort_keys=False))
                    job={"id":f"{family}_weight{weight}_{arm}_seed{seed}","family":family,"weight":weight,"arm":arm,"seed":seed,
                         "train_config":str(cfg_path),"eval_config":str(eval_path),"train_sha256":file_hash(train_file)}
                    inventory["jobs"].append(job)
                inventory["sources"].append({"pair":f"{family}_{arm}_seed{seed}","training_record":str(source_path),"training_record_sha256":file_hash(source_path),
                    "comparator_evaluation_record":str(evaluation_path),"original_config":source["config"],"resolved_config":cfg,
                    "validated_terminal_examples":checked[key],"adapter":str(adapter),
                    "historical_versions":source["versions"],"historical_data_hash_present":bool(source.get("train_file_sha256")),
                    "historical_code_hashes":source.get("code_sha256",{}),"primary_comparator":"fresh matched weight1"})
    inventory["code_sha256"]={str(p.relative_to(ROOT)):file_hash(p) for p in (
        ROOT/"scripts/tmlr_eos_control.py",ROOT/"scripts/train_terminal_eos.py",ROOT/"src/ila/terminal_eos_loss.py",
        ROOT/"scripts/train_sft.py",ROOT/"scripts/evaluate.py",ROOT/"scripts/analyze_tmlr_eos.py")}
    (CONTROL/"precheck.json").write_text(json.dumps(inventory,indent=2)+"\n")
    (CONTROL/"jobs.json").write_text(json.dumps(inventory["jobs"],indent=2)+"\n")
    for part in ("claims","done","failed","logs"):
        (CONTROL/part).mkdir()
    status("PRECHECK_PASSED")
    return inventory["jobs"]


def execute(job,gpu,smoke=False):
    env={**os.environ,"CUDA_VISIBLE_DEVICES":str(gpu),"HF_HUB_OFFLINE":"1","PYTHONPATH":"src",
         "ILA_GPU_MEM_GB":"34","ILA_VLLM_UTIL":"0.70","PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}
    cfg=load_config(job["train_config"])
    ecfg=load_config(job["eval_config"])
    seed=999 if smoke else job["seed"]
    commands=[]
    def run(command,name):
        log=CONTROL/"logs"/f"{job['id']}_{'smoke_' if smoke else ''}{name}.log"
        commands.append({"command":command,"log":str(log)})
        with log.open("w") as stream:
            subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    train=[PY,"scripts/train_terminal_eos.py","--config",job["train_config"],"--style",job["arm"],"--seed",str(seed)]
    if smoke:
        cfg["results_dir"]=f"results/{RUNNAME}/smoke_{job['family']}_w{job['weight']}/train"
        train += ["--set",f"results_dir={cfg['results_dir']}","train.stop_after_steps=1","train.snapshot_steps=[]"]
        ecfg["results_dir"]=f"results/{RUNNAME}/smoke_{job['family']}_w{job['weight']}/eval"
    run(train,"train")
    record_path=ROOT/cfg["results_dir"]/job["arm"]/f"seed{seed}/run_record.json"
    rec=json.loads(record_path.read_text())
    term=rec["terminal_eos_supervision"]
    if rec["global_steps"] != (1 if smoke else 60) or rec["n_truncated_examples"] or not rec["device"].startswith("cuda"):
        raise ValueError("training dose/device/truncation failure")
    if term["weight"] != job["weight"] or not term.get("model_accepts_loss_kwargs") or (job["weight"]!=1 and not term.get("received_accumulated_denominator")):
        raise ValueError("native accumulated-token normalization not verified")
    if rec["train_file_sha256"] != job["train_sha256"]:
        raise ValueError("trained data mismatch")
    if not rec.get("adapter_dir") or not (ROOT/rec["adapter_dir"]/"adapter_config.json").is_file():
        raise ValueError("missing trained adapter; refusing base-model fallback")
    style=f"lr1e5_step60__{job['arm']}"
    register(ecfg,style,seed,rec["adapter_dir"],str(record_path),1 if smoke else 60)
    for es,n in (("svamp",400),("aqua",254)):
        command=[EPY,"scripts/evaluate.py","--config",job["eval_config"],"--style",style,"--seed",str(seed),"--eval-set",es]
        if smoke:
            command += ["--limit","2","--set",f"results_dir={ecfg['results_dir']}"]
        run(command,es)
        path=ROOT/ecfg["results_dir"]/style/f"seed{seed}/eval"/es
        summary=json.loads((path/"summary.json").read_text())
        rows=read_jsonl(path/"outputs.jsonl")
        if summary["n"] != (2 if smoke else n) or summary["max_new_tokens"] != 6144 or summary["backend"] != "vllm":
            raise ValueError("evaluation config/size mismatch")
        if any(len(r["generated_token_ids"]) != r["n_gen_tokens"] for r in rows):
            raise ValueError("generated token count mismatch")
    return {"status":"completed","job":job,"physical_gpu":gpu,"commands":commands,"train_record":str(record_path),"smoke":smoke}


def worker(gpu):
    if gpu not in ALLOWED:raise ValueError("forbidden GPU")
    jobs=json.loads((CONTROL/"jobs.json").read_text())
    while True:
        remaining=[j for j in jobs if not (CONTROL/"claims"/j["id"]).exists()]
        if not remaining:return
        device,lease=acquire(gpu)
        if lease is None:return
        try:
            remaining=[j for j in jobs if not (CONTROL/"claims"/j["id"]).exists()]
            if not remaining:return
            job=remaining[0]
            try:(CONTROL/"claims"/job["id"]).mkdir()
            except FileExistsError:continue
            try:
                result=execute(job,device)
                (CONTROL/"done"/f"{job['id']}.json").write_text(json.dumps(result,indent=2))
            except Exception as error:
                (CONTROL/"failed"/f"{job['id']}.json").write_text(json.dumps({"job":job,"gpu":gpu,"error":repr(error)},indent=2))
        finally:lease.__exit__(None,None,None)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--worker",type=int);args=ap.parse_args()
    os.chdir(ROOT);os.environ.update(HF_HUB_OFFLINE="1",PYTHONPATH="src")
    if args.worker is not None:
        worker(args.worker);return
    if OUT.exists():raise SystemExit("EOS namespace exists; no automatic restart")
    try:
        jobs=prepare()
        for family in ("qwen","mistral"):
            for weight in (1,10):
                status(f"GPU_SMOKE_{family.upper()}_WEIGHT{weight}")
                job=next(j for j in jobs if j["family"]==family and j["weight"]==weight and j["arm"]=="armA_random" and j["seed"]==0)
                gpu,lease=acquire()
                try:result=execute(job,gpu,smoke=True)
                finally:lease.__exit__(None,None,None)
                (CONTROL/f"{family}_weight{weight}_smoke.json").write_text(json.dumps(result,indent=2))
        status("SMOKES_PASSED_RUNNING_36_TRAIN_AND_72_EVAL_QUEUE")
        processes=[];streams=[]
        for gpu in sorted(ALLOWED):
            stream=(CONTROL/f"worker_gpu{gpu}.log").open("w");streams.append(stream)
            processes.append(subprocess.Popen([PY,"scripts/tmlr_eos_control.py","--worker",str(gpu)],stdout=stream,stderr=subprocess.STDOUT))
        codes=[p.wait() for p in processes]
        for stream in streams:stream.close()
        if any(codes) or list((CONTROL/"failed").glob("*.json")) or len(list((CONTROL/"done").glob("*.json")))!=36:
            raise ValueError("incomplete/failed matrix; retain logs")
        status("FULL_MATRIX_DONE_VERIFYING_AND_ANALYZING")
        for family in ("qwen","mistral"):
            for weight in (1,10):
                subprocess.run([PY,"scripts/analyze_tmlr_external.py","--family",family,"--root",str(OUT/family/f"weight{weight}"/"eval"),
                    "--baseline-root",f"results/tmlr_external_{family}","--recipes","lr1e5_step60"],check=True)
        subprocess.run([PY,"scripts/analyze_tmlr_eos.py"],check=True)
        status("COMPLETED_VERIFIED_EOS_CONTROL_ANALYSIS_READY")
    except Exception as error:
        if CONTROL.exists():
            (CONTROL/"failure.json").write_text(json.dumps({"error":repr(error)},indent=2))
            status("FAILED_RETAINED_ARTIFACTS_REVIEW_REQUIRED")
        raise


if __name__=="__main__":main()
