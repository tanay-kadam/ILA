"""Supervised fine-tuning on one reasoning style (LoRA or full), completion-only loss.

Writes to <results_dir>/<style>/seed<k>/:
  model/            merged full-weight model (what evaluation loads)
  adapter/          LoRA adapter (LoRA runs only)
  train_log.json    per-step loss history
  run_record.json   command, resolved config, seed, device, versions, status
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import sys
import time
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
import transformers  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from ila.config import load_config  # noqa: E402
from ila.textio import build_chat_user_content, build_prompt, read_jsonl  # noqa: E402


class AdapterSnapshotCallback(TrainerCallback):
    """Save lightweight LoRA adapters at prospectively selected optimizer steps."""

    def __init__(self, run_dir: Path, steps: list[int]):
        self.run_dir = run_dir
        self.steps = {int(step) for step in steps}
        self.saved: list[int] = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        if step not in self.steps or step in self.saved or not state.is_world_process_zero:
            return control
        if model is None or not hasattr(model, "save_pretrained"):
            raise RuntimeError(f"cannot save requested adapter snapshot at step {step}")
        out = self.run_dir / "trajectory_adapters" / f"step{step:03d}"
        out.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(out)
        self.saved.append(step)
        manifest = {
            "requested_steps": sorted(self.steps),
            "saved_steps": sorted(self.saved),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (self.run_dir / "trajectory_adapters" / "manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        print(f"SNAPSHOT: step={step} adapter={out}", flush=True)
        return control


class StopAfterStepsCallback(TrainerCallback):
    """Stop optimization early without changing the scheduler's planned horizon."""

    def __init__(self, stop_after_steps: int):
        self.stop_after_steps = int(stop_after_steps)

    def on_step_end(self, args, state, control, **kwargs):
        if int(state.global_step) >= self.stop_after_steps:
            control.should_training_stop = True
        return control


def chat_template_ids(tok, messages: list[dict], add_generation_prompt: bool) -> list[int]:
    """Normalize Transformers 4/5 chat-template return types to one token-id list."""
    encoded = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError(f"expected one chat example, got batch of {len(encoded)}")
        encoded = encoded[0]
    return list(encoded)


class SFTDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        tok,
        max_len: int,
        template: str | None = None,
        canonical_chat: bool = False,
    ):
        self.items = []
        self.n_truncated = 0
        for r in rows:
            if canonical_chat:
                user = {"role": "user", "content": build_chat_user_content(r["question"], template)}
                p_ids = chat_template_ids(tok, [user], add_generation_prompt=True)
                ids = chat_template_ids(
                    tok,
                    [user, {"role": "assistant", "content": r["target"]}],
                    add_generation_prompt=False,
                )
                if ids[: len(p_ids)] != p_ids:
                    common = next(
                        (i for i, (a, b) in enumerate(zip(p_ids, ids)) if a != b),
                        min(len(p_ids), len(ids)),
                    )
                    raise ValueError(
                        "chat-template assistant prefix differs between prompt and full example: "
                        f"prompt_len={len(p_ids)} full_len={len(ids)} first_diff={common} "
                        f"prompt_tail={p_ids[max(0, common - 4):common + 6]} "
                        f"full_tail={ids[max(0, common - 4):common + 6]}"
                    )
                labels = [-100] * len(p_ids) + ids[len(p_ids) :]
            else:
                p_ids = tok(build_prompt(r["question"], template), add_special_tokens=False)["input_ids"]
                t_ids = tok(r["target"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
                ids = p_ids + t_ids
                labels = [-100] * len(p_ids) + t_ids
            if len(ids) > max_len:
                self.n_truncated += 1
                ids, labels = ids[:max_len], labels[:max_len]
            self.items.append({"input_ids": ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


def make_collator(pad_id: int):
    def collate(batch: list[dict]) -> dict:
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            labels[i, :L] = torch.tensor(b["labels"])
            mask[i, :L] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}

    return collate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--style", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--allow-cpu", action="store_true", help="only for smoke tests")
    ap.add_argument("--force", action="store_true", help="overwrite an existing run directory")
    ap.add_argument(
        "--chat-template",
        action="store_true",
        help="use canonical user/assistant chat serialization for SFT",
    )
    ap.add_argument(
        "--smoke-only",
        action="store_true",
        help="run at most five steps and record success without saving/merging a checkpoint",
    )
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    tcfg = cfg["train"]
    use_chat = args.chat_template or cfg.get("prompt_mode") == "canonical_chat"
    if args.smoke_only and not (0 < int(tcfg["max_steps"]) <= 5):
        raise SystemExit("--smoke-only requires train.max_steps between 1 and 5")

    run_dir = Path(cfg["results_dir"]) / args.style / f"seed{args.seed}"
    record_path = run_dir / "run_record.json"
    if record_path.exists() and not args.force:
        prev = json.loads(record_path.read_text())
        if prev.get("status") == "completed":
            if prev.get("config") != cfg:
                raise SystemExit("completed training config differs; preserve it and use a new output directory")
            data_path = Path(cfg["data_dir"]) / f"train_{args.style}.jsonl"
            if prev.get("train_file_sha256") and prev["train_file_sha256"] != hashlib.sha256(data_path.read_bytes()).hexdigest():
                raise SystemExit("completed training data fingerprint differs")
            print(f"SKIP: {run_dir} already completed")
            return
        raise SystemExit(f"{run_dir} has a non-completed run ({prev.get('status')}); inspect or pass --force")
    run_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name()})"
    elif args.allow_cpu:
        device = "cpu"
    else:
        raise SystemExit("CUDA not available; refusing to silently train on CPU (use --allow-cpu for smoke tests)")

    mem_cap_gb = os.environ.get("ILA_GPU_MEM_GB")
    if mem_cap_gb and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(mem_cap_gb) * 2**30 / total))

    record = {
        "status": "running",
        "command": " ".join(sys.argv),
        "style": args.style,
        "seed": args.seed,
        "model": cfg["model"],
        "method": tcfg["method"],
        "prompt_mode": "canonical_chat" if use_chat else "plain_text",
        "train_file": str(Path(cfg["data_dir"]) / f"train_{args.style}.jsonl"),
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "config": cfg,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    record["train_file_sha256"] = hashlib.sha256(Path(record["train_file"]).read_bytes()).hexdigest()
    record["code_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__), Path(__file__).resolve().parents[1] / "src/ila/textio.py",
        Path(__file__).resolve().parents[1] / "src/ila/config.py")}
    record_path.write_text(json.dumps(record, indent=2))

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    rows = read_jsonl(record["train_file"])
    ds = SFTDataset(
        rows,
        tok,
        tcfg["max_len"],
        cfg.get("prompt_template"),
        canonical_chat=use_chat,
    )
    record["n_train"] = len(ds)
    record["n_truncated_examples"] = ds.n_truncated
    record["n_target_tokens"] = sum(sum(l != -100 for l in it["labels"]) for it in ds.items)
    snapshot_steps = sorted({int(step) for step in tcfg.get("snapshot_steps", [])})
    stop_after_steps = int(tcfg.get("stop_after_steps", -1))
    if snapshot_steps and tcfg["method"] != "lora":
        raise SystemExit("train.snapshot_steps is currently supported only for LoRA runs")
    if any(step <= 0 for step in snapshot_steps):
        raise SystemExit("train.snapshot_steps must contain positive integers")
    if int(tcfg["max_steps"]) > 0 and any(step > int(tcfg["max_steps"]) for step in snapshot_steps):
        raise SystemExit("train.snapshot_steps cannot exceed train.max_steps")
    if stop_after_steps == 0 or stop_after_steps < -1:
        raise SystemExit("train.stop_after_steps must be -1 or a positive integer")
    if stop_after_steps > 0 and int(tcfg["max_steps"]) > 0 and stop_after_steps > int(tcfg["max_steps"]):
        raise SystemExit("train.stop_after_steps cannot exceed train.max_steps")
    record["snapshot_steps"] = snapshot_steps
    record["planned_max_steps"] = int(tcfg["max_steps"])
    record["stop_after_steps"] = stop_after_steps

    use_bf16 = bool(tcfg["bf16"]) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16 if use_bf16 else torch.float32
    )
    if tcfg["method"] == "lora":
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=tcfg["lora_r"],
                lora_alpha=tcfg["lora_alpha"],
                lora_dropout=tcfg["lora_dropout"],
                target_modules=tcfg["target_modules"],
                task_type="CAUSAL_LM",
            ),
        )
        record["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif tcfg["method"] != "full":
        raise SystemExit(f"unknown train.method {tcfg['method']}")

    # transformers>=5 folds warmup_ratio into a fractional warmup_steps.
    if "warmup_ratio" in inspect.signature(TrainingArguments.__init__).parameters:
        warmup = {"warmup_ratio": tcfg["warmup_ratio"]}
    else:
        warmup = {"warmup_steps": float(tcfg["warmup_ratio"])}
    targs = TrainingArguments(
        output_dir=str(run_dir / "trainer"),
        per_device_train_batch_size=tcfg["per_device_batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum"],
        learning_rate=tcfg["learning_rate"],
        num_train_epochs=tcfg["epochs"],
        max_steps=tcfg["max_steps"],
        **warmup,
        lr_scheduler_type=tcfg["lr_scheduler"],
        weight_decay=tcfg["weight_decay"],
        bf16=use_bf16,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", False)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    callbacks = [AdapterSnapshotCallback(run_dir, snapshot_steps)] if snapshot_steps else []
    if stop_after_steps > 0:
        callbacks.append(StopAfterStepsCallback(stop_after_steps))
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=make_collator(tok.pad_token_id),
        callbacks=callbacks,
    )
    try:
        result = trainer.train()
    except Exception as e:
        record.update(status="failed", error=repr(e))
        record_path.write_text(json.dumps(record, indent=2))
        raise

    (run_dir / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    if args.smoke_only:
        record.update(
            status="smoke_completed",
            finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
            global_steps=trainer.state.global_step,
            final_train_loss=result.training_loss,
            peak_gpu_mem_gb=(torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else None,
            gpu_mem_cap_gb=float(mem_cap_gb) if mem_cap_gb else None,
            model_dir=None,
        )
        record_path.write_text(json.dumps(record, indent=2))
        print(f"SMOKE DONE: {run_dir} steps={trainer.state.global_step} loss={result.training_loss:.4f}")
        return
    if tcfg["method"] == "lora":
        model.save_pretrained(run_dir / "adapter")
        if bool(tcfg.get("save_merged_model", True)):
            model = model.merge_and_unload()
            model.save_pretrained(run_dir / "model", safe_serialization=True)
            tok.save_pretrained(run_dir / "model")
            model_dir = str(run_dir / "model")
            adapter_dir = None
        else:
            model_dir = cfg["model"]
            adapter_dir = str(run_dir / "adapter")
    else:
        model.save_pretrained(run_dir / "model", safe_serialization=True)
        tok.save_pretrained(run_dir / "model")
        model_dir = str(run_dir / "model")
        adapter_dir = None

    record.update(
        status="completed",
        finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
        global_steps=trainer.state.global_step,
        final_train_loss=result.training_loss,
        peak_gpu_mem_gb=(torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else None,
        gpu_mem_cap_gb=float(mem_cap_gb) if mem_cap_gb else None,
        model_dir=model_dir,
        adapter_dir=adapter_dir,
    )
    record_path.write_text(json.dumps(record, indent=2))
    print(f"DONE: {run_dir} steps={trainer.state.global_step} loss={result.training_loss:.4f}")


if __name__ == "__main__":
    main()
