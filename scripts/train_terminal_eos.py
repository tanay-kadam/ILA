"""Run unchanged SFT setup with a narrowly scoped terminal-EOS loss intervention."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import train_sft as native
from ila.config import load_config
from ila.terminal_eos_loss import terminal_eos_extra, validate_terminal_dataset


def trainer_class(weight, eos_token_id, diagnostics, tokenizer=None):
    class TerminalTrainer(native.Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.label_smoother is not None or self.compute_loss_func is not None:
                raise ValueError("unsupported non-native loss path")
            if self.accelerator.num_processes != 1 or self.args.n_gpu > 1:
                raise ValueError("EOS control requires one visible GPU per process")
            self.terminal_suffix_ids=set()
            diagnostics["weighted_terminal_examples"] = validate_terminal_dataset(self.train_dataset, eos_token_id,tokenizer,self.terminal_suffix_ids)
            diagnostics["allowed_serialized_suffix_ids"]=sorted(self.terminal_suffix_ids)
            diagnostics["model_accepts_loss_kwargs"] = self.model_accepts_loss_kwargs
            diagnostics["effective_grad_accum"] = self.args.gradient_accumulation_steps

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if weight == 1:
                return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
            labels = inputs["labels"]
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
            denominator = num_items_in_batch if self.model_accepts_loss_kwargs else None
            extra = terminal_eos_extra(outputs["logits"], labels, eos_token_id, denominator,self.terminal_suffix_ids)
            loss = loss + (weight-1)*extra
            diagnostics["loss_calls"] = diagnostics.get("loss_calls", 0)+1
            diagnostics["received_accumulated_denominator"] = num_items_in_batch is not None
            return (loss, outputs) if return_outputs else loss
    return TerminalTrainer


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", required=True); ap.add_argument("--style", required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--set", nargs="*", default=[])
    args, _ = ap.parse_known_args()
    cfg = load_config(args.config, args.set)
    weight = float(cfg["train"]["terminal_eos_weight"])
    if weight not in (1.0, 10.0):
        raise ValueError("only fixed weights 1 or 10 are authorized by this protocol")
    path = Path(cfg["results_dir"])/args.style/f"seed{args.seed}"
    if path.exists():
        raise ValueError("control run already exists; preserved without overwrite/restart")
    path.mkdir(parents=True)
    tok = native.AutoTokenizer.from_pretrained(cfg["model"])
    info = {"weight": weight, "eos_token_id": tok.eos_token_id,
            "formula": "native loss + (weight-1)*terminal CE / native token denominator",
            "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (Path(__file__), ROOT/"src/ila/terminal_eos_loss.py", ROOT/"scripts/train_sft.py")},
            "status": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    native.Trainer = trainer_class(weight, tok.eos_token_id, info,tok)
    control_path = path/"terminal_eos_record.json"
    control_path.write_text(json.dumps(info, indent=2))
    try:
        native.main()
        info["status"] = "completed"
    except Exception as error:
        info.update(status="failed", error=repr(error))
        raise
    finally:
        info["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        control_path.write_text(json.dumps(info, indent=2))
        record_path = path/"run_record.json"
        if record_path.exists():
            record = json.loads(record_path.read_text())
            record["terminal_eos_supervision"] = info
            record_path.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
