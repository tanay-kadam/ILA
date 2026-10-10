"""Generate fixed evaluation/training jobs; no outcome-dependent selection."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from ila.config import load_config


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--family",choices=("qwen","mistral"),required=True);ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    cfg=f"configs/tmlr_external_{args.family}"+("_resolved.yaml" if args.family=="mistral" else ".yaml")
    jobs=[{"id":f"{args.family}_base","mode":"eval","config":cfg,"style":"_base","seed":0,"base":True}]
    if args.family=="qwen":
        recipes={"lr1e5_step15":"stage1c_matrix","lr1e5_step60":"stage1c_dose60","lr3e6_step60":"stage1c_final_stability"}
        for recipe,folder in recipes.items():
            for arm in ("armA_high","armA_low","armA_random"):
                for seed in range(3):
                    source=Path("results")/folder/arm/f"seed{seed}/run_record.json"
                    rec=json.loads(source.read_text())
                    adapter=rec["adapter_dir"]
                    if not Path(adapter,"adapter_config.json").is_file():raise RuntimeError(f"missing {adapter}")
                    jobs.append({"id":f"{recipe}_{arm}_seed{seed}","mode":"eval","config":cfg,
                                 "style":f"{recipe}__{arm}","seed":seed,"adapter":adapter,"source":str(source),"step":15 if "step15" in recipe else 60})
    else:
        for lr in ("1e5","3e6"):
            for arm in ("armA_high","armA_low","armA_random"):
                for seed in range(3):
                    jobs.append({"id":f"mistral_lr{lr}_{arm}_seed{seed}","mode":"train","config":f"configs/tmlr_mistral_lr{lr}_resolved.yaml","arm":arm,"seed":seed,"lr":lr})
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(jobs,indent=2)+"\n")
    print(f"{args.family}: {len(jobs)} jobs")


if __name__=="__main__":main()
