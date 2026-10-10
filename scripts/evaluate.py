"""Greedy-decode an eval set and record per-question inference cost and correctness.

Writes to <run_dir>/eval/<set>/:
  outputs.jsonl   uid, output, n_gen_tokens, truncated, pred, correct, markers
  summary.json    aggregate metrics + provenance

n_gen_tokens counts generated tokens excluding the terminating EOS. Truncated outputs
(no EOS within max_new_tokens) are counted at the cap, so means are lower bounds when
truncation is non-zero; truncation rate is always reported alongside.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ila.config import load_config  # noqa: E402
from ila.features import EVAL_MARKERS  # noqa: E402
from ila.styles import MARKERS as STYLE_MARKERS  # noqa: E402
from ila.textio import (  # noqa: E402
    answers_match,
    answers_match_choice,
    build_chat_user_content,
    build_prompt,
    extract_answer,
    read_jsonl,
    write_jsonl,
)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def evaluation_fingerprint(cfg, record, data_file, style, seed, limit, chat):
    adapter = record.get("adapter_dir")
    adapter_hashes = {} if not adapter else {
        p.name: file_hash(p) for p in sorted(Path(adapter).glob("*"))
        if p.is_file() and (p.suffix in (".safetensors", ".bin") or p.name == "adapter_config.json")
    }
    spec = {"config": cfg, "model": record["model_dir"], "adapter": adapter,
            "adapter_hashes": adapter_hashes, "data_sha256": file_hash(data_file),
            "style": style, "seed": seed, "limit": limit, "chat": chat,
            "evaluator_sha256": file_hash(__file__)}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def check_cached_summary(summary, fingerprint):
    if summary.get("evaluation_fingerprint") != fingerprint:
        raise SystemExit("existing evaluation lacks matching provenance; preserved without overwrite; use a new output directory or explicitly --force")


def load_hf_model(model_dir, dtype, device, adapter_dir=None):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    return model.to(device).eval()


def build_eval_prompts(rows: list[dict], template: str | None, chat_tokenizer=None) -> list[str]:
    """Build either the preregistered plain prompts or canonical single-turn chat prompts.

    In chat mode the trailing ``Answer:`` cue is removed because the tokenizer's generation
    prompt already opens the assistant turn. The semantic instruction is otherwise unchanged.
    """
    plain = [build_prompt(r["question"], template) for r in rows]
    if chat_tokenizer is None:
        return plain
    user_messages = [build_chat_user_content(r["question"], template) for r in rows]
    return [
        chat_tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for content in user_messages
    ]


def generate_vllm(
    model_dir: str,
    prompts: list[str],
    max_new_tokens: int,
    tokenizer: str | None = None,
    adapter_dir: str | None = None,
    max_lora_rank: int = 64,
) -> list[tuple[str, int, bool, list[int]]]:
    from vllm import LLM, SamplingParams

    util = float(os.environ.get("ILA_VLLM_UTIL", "0.85"))
    # Tokenizers saved by transformers 5 are unreadable by the 4.x copy in the vLLM venv; training
    # never changes the tokenizer, so load the base model's.
    lora_args = {"enable_lora": True, "max_lora_rank": max_lora_rank} if adapter_dir else {}
    llm = LLM(model=model_dir, tokenizer=tokenizer or model_dir, dtype="bfloat16", seed=0,
              gpu_memory_utilization=util,
              max_model_len=max_new_tokens + 2048, enforce_eager=False, **lora_args)
    tok = llm.get_tokenizer()
    stop_ids = sorted({i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None})
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, stop_token_ids=stop_ids)
    if adapter_dir:
        from vllm.lora.request import LoRARequest

        outs = llm.generate(prompts, sp, lora_request=LoRARequest("stage1c", 1, adapter_dir))
    else:
        outs = llm.generate(prompts, sp)
    res = []
    for o in outs:
        c = o.outputs[0]
        ids = list(c.token_ids)
        if ids and ids[-1] in stop_ids:
            ids = ids[:-1]
        res.append((c.text, len(ids), c.finish_reason == "length", ids))
    return res


def generate_hf(model_dir: str, prompts: list[str], max_new_tokens: int, batch_size: int, allow_cpu: bool, adapter_dir=None, tokenizer=None):
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available() and not allow_cpu:
        raise SystemExit("CUDA not available; refusing to silently evaluate on CPU")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cap_gb = os.environ.get("ILA_GPU_MEM_GB")
    if cap_gb and dev == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(cap_gb) * 2**30 / total))
    tok = AutoTokenizer.from_pretrained(tokenizer or model_dir, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    model = load_hf_model(model_dir, dtype, dev, adapter_dir)
    stop_ids = sorted({tok.eos_token_id, tok.pad_token_id})
    res = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=stop_ids,
                pad_token_id=tok.pad_token_id,
            )
        for row in gen[:, enc["input_ids"].shape[1] :].tolist():
            cut = [row.index(s) for s in stop_ids if s in row]
            if cut:
                ids, trunc = row[: min(cut)], False
            else:
                ids, trunc = row, True
            res.append((tok.decode(ids, skip_special_tokens=True), len(ids), trunc, ids))
    return res


def get_markers(ecfg: dict) -> dict[str, list[str]]:
    return EVAL_MARKERS if ecfg.get("markers") == "reflection_restart" else STYLE_MARKERS


def marker_counts(text: str, markers: dict[str, list[str]]) -> dict[str, int]:
    return {name: sum(text.count(m) for m in ms) for name, ms in markers.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--style", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--base", action="store_true", help="evaluate the untuned cfg model as style '_base'")
    ap.add_argument(
        "--chat-template",
        action="store_true",
        help="wrap the semantic prompt in the base tokenizer's canonical chat template",
    )
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    ecfg = cfg["eval"]
    scfg = ecfg["sets"][args.eval_set]
    markers = get_markers(ecfg)
    template = cfg.get("prompt_template")
    fmt = scfg.get("answer_format", cfg.get("answer_format", "answer_is"))

    run_dir = Path(cfg["results_dir"]) / args.style / f"seed{args.seed}"
    if args.base:
        rec = {"status": "completed", "model_dir": cfg["model"]}
    else:
        rec = json.loads((run_dir / "run_record.json").read_text())
    if rec.get("status") != "completed":
        raise SystemExit(f"training run {run_dir} is not completed ({rec.get('status')})")
    out_dir = run_dir / "eval" / args.eval_set
    data_file = Path(cfg["data_dir"]) / scfg["file"]
    use_chat = args.chat_template or cfg.get("prompt_mode") == "canonical_chat"
    fingerprint = evaluation_fingerprint(cfg, rec, data_file, args.style, args.seed, args.limit, use_chat)
    if (out_dir / "summary.json").exists() and not args.force:
        check_cached_summary(json.loads((out_dir / "summary.json").read_text()), fingerprint)
        print(f"SKIP: {out_dir} already evaluated")
        return

    rows = read_jsonl(data_file)
    if args.limit:
        rows = rows[: args.limit]
    chat_tok = None
    use_chat = args.chat_template or cfg.get("prompt_mode") == "canonical_chat"
    if use_chat:
        from transformers import AutoTokenizer

        chat_tok = AutoTokenizer.from_pretrained(cfg["model"])
    prompts = build_eval_prompts(rows, template, chat_tok)
    model_dir = rec["model_dir"]
    adapter_dir = rec.get("adapter_dir")
    t0 = time.time()
    if ecfg["backend"] == "vllm":
        gens = generate_vllm(
            model_dir,
            prompts,
            scfg["max_new_tokens"],
            cfg["model"],
            adapter_dir=adapter_dir,
            max_lora_rank=int(rec.get("max_lora_rank", cfg.get("train", {}).get("lora_r", 64))),
        )
    else:
        gens = generate_hf(model_dir, prompts, scfg["max_new_tokens"], ecfg["hf_batch_size"], args.allow_cpu,
                           adapter_dir=adapter_dir, tokenizer=cfg["model"])
    wall = time.time() - t0

    outs = []
    if len(gens) != len(rows):
        raise SystemExit("generation count mismatch; refusing partial metrics")
    for r, (text, n, trunc, token_ids) in zip(rows, gens):
        pred = extract_answer(text, fmt)
        outs.append(
            {
                "uid": r["uid"],
                "output": text,
                "n_gen_tokens": n,
                "generated_token_ids": token_ids,
                "truncated": trunc,
                "pred": pred,
                "gold": r["gold"],
                "correct": answers_match_choice(pred, r["gold"]) if fmt == "boxed_choice" else answers_match(pred, r["gold"]),
                "markers": marker_counts(text, markers),
            }
        )
    write_jsonl(out_dir / "outputs.jsonl", outs)

    toks = [o["n_gen_tokens"] for o in outs]
    n_correct = sum(o["correct"] for o in outs)
    summary = {
        "evaluation_fingerprint": fingerprint,
        "versions": {name: package_version(name) for name in ("torch", "transformers", "peft")},
        "backend_version": package_version("vllm" if ecfg["backend"] == "vllm" else "transformers"),
        "vllm_memory_utilization": float(os.environ.get("ILA_VLLM_UTIL", "0.85")) if ecfg["backend"] == "vllm" else None,
        "style": args.style,
        "seed": args.seed,
        "eval_set": args.eval_set,
        "model_dir": model_dir,
        "adapter_dir": adapter_dir,
        "backend": ecfg["backend"],
        "decoding": "greedy",
        "prompt_mode": "canonical_chat" if use_chat else "plain_text",
        "answer_format": fmt,
        "max_new_tokens": scfg["max_new_tokens"],
        "n": len(outs),
        "accuracy": n_correct / len(outs),
        "mean_tokens": statistics.mean(toks),
        "median_tokens": statistics.median(toks),
        "p90_tokens": sorted(toks)[int(0.9 * (len(toks) - 1))],
        "truncation_rate": sum(o["truncated"] for o in outs) / len(outs),
        "tokens_per_correct": sum(toks) / n_correct if n_correct else None,
        "marker_rate": {s: sum(o["markers"][s] > 0 for o in outs) / len(outs) for s in markers},
        "wall_seconds": wall,
        "gpu_mem_cap_gb": float(os.environ["ILA_GPU_MEM_GB"]) if os.environ.get("ILA_GPU_MEM_GB") else None,
        "hf_batch_size": ecfg["hf_batch_size"],
        "limit": args.limit,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("style", "seed", "eval_set", "accuracy", "mean_tokens", "truncation_rate")}))


if __name__ == "__main__":
    main()
