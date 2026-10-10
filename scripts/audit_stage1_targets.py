"""Read-only audit of Stage 1 natural SFT targets and canonical terminal tokens."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_stage1_outputs import boxed_spans, loop_metrics  # noqa: E402
from ila.config import load_config  # noqa: E402
from ila.textio import build_chat_user_content, read_jsonl  # noqa: E402


def dist(xs: list[int]) -> dict:
    return {
        "n": len(xs),
        "mean": statistics.mean(xs) if xs else None,
        "median": statistics.median(xs) if xs else None,
        "max": max(xs) if xs else None,
    }


def ids_list(encoded) -> list[int]:
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError(f"expected one chat example, got batch of {len(encoded)}")
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--style", default="armA_random")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    rows = read_jsonl(Path(cfg["data_dir"]) / f"train_{args.style}.jsonl")
    token_lengths, post_box, first_box_tokens = [], [], []
    terminal_ids, terminal_sequences, eos_offsets = Counter(), Counter(), Counter()
    loop_count, boxed_count, eos_near_end = 0, 0, 0
    for row in rows:
        target = row["target"]
        target_ids = tok(target, add_special_tokens=False)["input_ids"]
        token_lengths.append(len(target_ids))
        spans = boxed_spans(target)
        boxed_count += bool(spans)
        if spans:
            prefix_n = len(tok(target[: spans[0]["end"]], add_special_tokens=False)["input_ids"])
            first_box_tokens.append(prefix_n)
            post_box.append(len(target_ids) - prefix_n)
        loop_count += loop_metrics(target)["likely_loop"]
        user = {"role": "user", "content": build_chat_user_content(row["question"], cfg.get("prompt_template"))}
        full = ids_list(tok.apply_chat_template(
            [user, {"role": "assistant", "content": target}],
            tokenize=True,
            add_generation_prompt=False,
        ))
        last = full[-1]
        terminal_ids[last] += 1
        terminal_sequences[tuple(full[-4:])] += 1
        reverse_offset = next(
            (offset for offset, token_id in enumerate(reversed(full)) if token_id == tok.eos_token_id),
            None,
        )
        if reverse_offset is not None:
            eos_offsets[reverse_offset] += 1
            eos_near_end += reverse_offset < 4
    report = {
        "n": len(rows),
        "model": cfg["model"],
        "boxed_rate": boxed_count / len(rows),
        "likely_loop_rate": loop_count / len(rows),
        "target_tokens": dist(token_lengths),
        "first_box_tokens": dist(first_box_tokens),
        "post_first_box_tokens": dist(post_box),
        "terminal_token_counts": {str(k): v for k, v in terminal_ids.most_common()},
        "terminal_four_token_sequences": {
            ",".join(map(str, k)): v for k, v in terminal_sequences.most_common()
        },
        "eos_token_id": tok.eos_token_id,
        "eos_reverse_offset_counts": {str(k): v for k, v in sorted(eos_offsets.items())},
        "eos_within_last_four_rate": eos_near_end / len(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
