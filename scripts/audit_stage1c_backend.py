"""Bounded tokenizer and HF-adapter diagnostic; leaves experiment outputs intact."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ila.config import load_config
from ila.textio import build_chat_user_content, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("tokenizer", "hf"), default="tokenizer")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--max-tokens", type=int, default=768)
    args = ap.parse_args()
    cfg = load_config("configs/stage1c_dose60.yaml")
    from transformers import AutoTokenizer
    import transformers
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    data = read_jsonl("data/stage1/eval_gsm8k_test.jsonl")
    row = next(r for r in data if r["uid"] == "gsm8k-test-4")
    message = [{"role": "user", "content": build_chat_user_content(row["question"], cfg["prompt_template"])}]
    prompt = tok.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    report = {"transformers": transformers.__version__, "eos_id": tok.eos_token_id, "pad_id": tok.pad_token_id,
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
              "prompt_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(), "prompt_tail_ids": ids[-12:]}
    if args.mode == "tokenizer":
        terminals = {}
        for name in ("armA_high", "armA_low", "armA_random"):
            train = read_jsonl(f"data/stage1/train_{name}.jsonl")
            sample = train[:16]
            tails = []
            for r in sample:
                user = {"role": "user", "content": build_chat_user_content(r["question"], cfg["prompt_template"])}
                full = tok.apply_chat_template([user, {"role": "assistant", "content": r["target"]}], tokenize=False, add_generation_prompt=False)
                tail = tok(full, add_special_tokens=False)["input_ids"][-4:]
                tails.append(tail)
            terminals[name] = {"sample_n": len(sample), "eos_in_tail": sum(tok.eos_token_id in t for t in tails), "tails": tails}
            if transformers.__version__.startswith("5."):
                from train_sft import SFTDataset, chat_template_ids
                ds = SFTDataset(sample, tok, cfg["train"]["max_len"], cfg["prompt_template"], canonical_chat=True)
                terminals[name]["sample_truncated"] = ds.n_truncated
                terminals[name]["supervised_eos_examples"] = sum(tok.eos_token_id in it["labels"][-4:] for it in ds.items)
                terminals[name]["prompt_mask_correct_examples"] = sum(
                    all(label == -100 for label in it["labels"][:len(chat_template_ids(tok,
                        [{"role": "user", "content": build_chat_user_content(r["question"], cfg["prompt_template"])}], True))])
                    for r, it in zip(sample, ds.items))
        report["sampled_training_terminals"] = terminals
    else:
        import torch
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        if not torch.cuda.is_available():
            raise SystemExit("CUDA required")
        adapter = "results/stage1c_dose60/armA_high/seed0/adapter"
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, adapter).to("cuda").eval()
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            gen = model.generate(**enc, do_sample=False, max_new_tokens=args.max_tokens,
                                 eos_token_id=sorted({tok.eos_token_id, tok.pad_token_id}), pad_token_id=tok.pad_token_id)
        out_ids = gen[0, enc["input_ids"].shape[1]:].tolist()
        text = tok.decode(out_ids, skip_special_tokens=True)
        old = next(r for r in read_jsonl("results/stage1c_dose60/armA_high/seed0/eval/gsm8k_test/outputs.jsonl") if r["uid"] == row["uid"])
        old_ids = tok(old["output"], add_special_tokens=False)["input_ids"]
        common = next((i for i, (a, b) in enumerate(zip(out_ids, old_ids)) if a != b), min(len(out_ids), len(old_ids)))
        from diagnose_stage1_outputs import loop_metrics
        report.update(adapter=adapter, device=torch.cuda.get_device_name(), uid=row["uid"],
                      generated_tokens=len(out_ids), eos_generated=tok.eos_token_id in out_ids,
                      common_prefix_tokens_with_vllm=common, hf_output=text,
                      hf_loop_metrics=loop_metrics(text), vllm_original_tokens=old["n_gen_tokens"],
                      vllm_original_truncated=old["truncated"])
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("sampled_training_terminals", "hf_output")}, indent=2))


if __name__ == "__main__":
    main()
