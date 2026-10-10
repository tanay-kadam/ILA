"""Pin/download one second-family model, validate serialization, resolve configs."""
from __future__ import annotations
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from ila.config import load_config
from ila.textio import read_jsonl


def main():
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer
    from train_sft import SFTDataset
    model_id="mistralai/Mistral-7B-Instruct-v0.3"
    out=ROOT/"results/tmlr_mistral_precheck"; out.mkdir(exist_ok=True)
    manifest=out/"model_manifest.json"
    if manifest.exists():
        info=json.loads(manifest.read_text()); revision=info["revision"]
    else:
        revision=HfApi().model_info(model_id).sha
        manifest.write_text(json.dumps({"model_id":model_id,"revision":revision},indent=2))
    print(f"FROZEN_MODEL {model_id} revision={revision}",flush=True)
    path=snapshot_download(model_id,revision=revision,allow_patterns=["config.json","generation_config.json","tokenizer*","special_tokens_map.json"])
    tokenizer=AutoTokenizer.from_pretrained(path)
    cfg=load_config(ROOT/"configs/tmlr_mistral_lr1e5.yaml")
    checks={}
    for arm in ("armA_high","armA_low","armA_random"):
        rows=read_jsonl(ROOT/"data/stage1"/f"train_{arm}.jsonl")
        ds=SFTDataset(rows,tokenizer,4096,cfg["prompt_template"],canonical_chat=True)
        eos=sum(tokenizer.eos_token_id in item["labels"][-4:] for item in ds.items)
        checks[arm]={"n":len(rows),"n_truncated":ds.n_truncated,"supervised_terminal_eos":eos}
        if ds.n_truncated or eos!=len(rows):
            (out/"serialization_failed.json").write_text(json.dumps(checks,indent=2))
            raise SystemExit("Mistral serialization/truncation precheck failed; family branch stopped")
    print("SERIALIZATION_PASSED_DOWNLOADING_WEIGHTS",flush=True)
    snapshot_download(model_id,revision=revision,allow_patterns=["config.json","generation_config.json","model*.safetensors","model*.json","tokenizer*","special_tokens_map.json"])
    for stem in ("tmlr_mistral_lr1e5","tmlr_mistral_lr3e6","tmlr_external_mistral"):
        resolved=load_config(ROOT/"configs"/f"{stem}.yaml")
        resolved.update(model=path,source_model_id=model_id,source_model_revision=revision)
        import yaml
        target=ROOT/"configs"/f"{stem}_resolved.yaml"
        target.write_text(yaml.safe_dump(resolved,sort_keys=False))
    info={"model_id":model_id,"revision":revision,"snapshot_path":path,"checks":checks}
    (out/"precheck.json").write_text(json.dumps(info,indent=2)+"\n")
    print(json.dumps(info,indent=2))


if __name__=="__main__":main()
