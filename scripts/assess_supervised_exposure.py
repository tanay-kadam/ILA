"""Measure canonical supervised lengths and fixed seedable sampler dose, CPU only."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from ila.config import load_config
from ila.textio import read_jsonl
from evaluate import file_hash


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise ValueError("report exists; preserved")
    import torch
    from transformers import AutoTokenizer,TrainingArguments,Trainer,GPT2Config,GPT2LMHeadModel,set_seed
    from train_sft import SFTDataset
    control=ROOT/"results/tmlr_eos_control_v3/controller"
    report={"families":{},"sampler_scope":"replay only when prepared Trainer sampler exposes deterministic initial_seed matching recorded data_seed; no guessed permutations",
            "protocol":"docs/tmlr_matching_feasibility_protocol.md"}
    for family in ("qwen","mistral"):
        banks={};lengths={};configs={};seeds={}
        for arm in ("armA_high","armA_low","armA_random"):
            cfg=load_config(control/f"{family}_weight1_{arm}_seed0_train.yaml");configs[arm]=cfg
            tok=AutoTokenizer.from_pretrained(cfg["model"])
            ds=SFTDataset(read_jsonl(ROOT/cfg["data_dir"]/f"train_{arm}.jsonl"),tok,cfg["train"]["max_len"],cfg["prompt_template"],canonical_chat=True)
            if ds.n_truncated:raise ValueError("unexpected truncation")
            values=np.array([sum(x!=-100 for x in item["labels"]) for item in ds.items])
            lengths[arm]=values
            record=json.loads((ROOT/cfg["results_dir"]/arm/"seed0/run_record.json").read_text())
            if int(values.sum())!=record["n_target_tokens"]:raise ValueError("canonical lengths disagree with completed training record")
            banks[arm]={"n":len(values),"mean":float(values.mean()),"std":float(values.std()),"total":int(values.sum())}
        for seed in range(3):
            cfg=configs["armA_random"];t=cfg["train"]
            set_seed(seed)
            model=GPT2LMHeadModel(GPT2Config(vocab_size=7,n_positions=8,n_embd=8,n_layer=1,n_head=1))
            args_cpu=TrainingArguments(output_dir=str(control/"unused_sampler_probe"),use_cpu=True,report_to=[],seed=seed,data_seed=seed,
                per_device_train_batch_size=t["per_device_batch_size"],gradient_accumulation_steps=t["grad_accum"])
            # Iterating the sampler, not the loader: no toy-model predictions/training.
            trainer=Trainer(model=model,args=args_cpu,train_dataset=list(range(len(lengths["armA_random"]))))
            loader=trainer.get_train_dataloader()
            sampler=loader.batch_sampler.sampler
            name=type(sampler).__name__;initial=getattr(sampler,"initial_seed",None)
            proven=name=="SeedableRandomSampler" and initial==seed
            result={"sampler_type":name,"initial_seed":initial,"recorded_data_seed":seed,"replay_supported":proven}
            if proven:
                indices=list(iter(sampler))[:960]
                result["n"]=len(indices);result["index_sha256"]=__import__("hashlib").sha256(json.dumps(indices).encode()).hexdigest()
                result["arms"]={arm:{"supervised_tokens":int(values[indices].sum()),"mean":float(values[indices].mean())} for arm,values in lengths.items()}
            seeds[str(seed)]=result
        comparisons={}
        base=banks["armA_random"]
        for arm in ("armA_high","armA_low"):
            current=banks[arm];pooled=np.sqrt((current["std"]**2+base["std"]**2)/2)
            comparisons[arm+"_vs_random"]={"relative_mean_difference":current["mean"]/base["mean"]-1,
                "standardized_mean_difference":(current["mean"]-base["mean"])/pooled}
        report["families"][family]={"banks":banks,"bank_comparisons":comparisons,"seed_dose":seeds,
            "residual_bank_imbalance_screen":any(abs(v["relative_mean_difference"])>.01 or abs(v["standardized_mean_difference"])>.1 for v in comparisons.values())}
    report["code_sha256"]=file_hash(Path(__file__))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({f:{"comparisons":v["bank_comparisons"],"replay_supported":all(s["replay_supported"] for s in v["seed_dose"].values())} for f,v in report["families"].items()}))


if __name__=="__main__":main()
