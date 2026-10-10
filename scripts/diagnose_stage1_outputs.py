"""Audit whether Stage 1 truncation occurs before or after a usable final answer.

This script is deliberately read-only with respect to evaluation artifacts. It reads one
or more ``outputs.jsonl`` files and writes a JSON report plus a compact Markdown review
packet. When ``--tokenizer`` is supplied, boxed-answer positions are measured exactly in
model tokens; otherwise positions are estimated by character offset for quick local use.

Example:

    python scripts/diagnose_stage1_outputs.py \
      --set gsm8k=results/stage1_q7/_base/seed0/eval/gsm8k_test/outputs.jsonl \
      --set math500=results/stage1_q7/_base/seed0/eval/math500/outputs.jsonl \
      --tokenizer Qwen/Qwen2.5-7B-Instruct \
      --output results/stage1_q7/diagnostic
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import statistics
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ila.textio import answers_match, read_jsonl  # noqa: E402


def boxed_spans(text: str) -> list[dict]:
    """Return every complete ``\\boxed{...}`` with brace-matched character spans."""
    out: list[dict] = []
    cursor = 0
    needle = "\\boxed{"
    while True:
        start = text.find(needle, cursor)
        if start < 0:
            return out
        i = start + len(needle)
        depth = 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            out.append({"content": text[start + len(needle) : i - 1], "start": start, "end": i})
            cursor = i
        else:
            # An incomplete box cannot contain a later complete top-level box.
            return out


def loop_metrics(text: str, suffix_words: int = 1024) -> dict:
    """Conservative, label-free repetition indicators for the tail of a completion."""
    words = re.findall(r"\S+", text)
    tail = words[-suffix_words:]
    if not tail:
        return {
            "tail_words": 0,
            "unique_4gram_ratio": 1.0,
            "max_4gram_share": 0.0,
            "compression_ratio": 1.0,
            "likely_loop": False,
        }
    grams = [tuple(tail[i : i + 4]) for i in range(max(0, len(tail) - 3))]
    counts = Counter(grams)
    unique_ratio = len(counts) / len(grams) if grams else 1.0
    max_share = max(counts.values()) / len(grams) if grams else 0.0
    raw = " ".join(tail).encode("utf-8", errors="replace")
    compression = len(zlib.compress(raw, level=9)) / len(raw) if raw else 1.0
    # These thresholds intentionally flag only strong repetition. The continuous metrics
    # and human-review packet remain authoritative for borderline cases.
    likely = len(tail) >= 128 and (
        unique_ratio < 0.25 or compression < 0.20 or max_share > 0.05
    )
    return {
        "tail_words": len(tail),
        "unique_4gram_ratio": unique_ratio,
        "max_4gram_share": max_share,
        "compression_ratio": compression,
        "likely_loop": likely,
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - pos) + xs[hi] * (pos - lo))


def distribution(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
    }


def make_position_counter(tokenizer_name: str | None) -> tuple[Callable[[str, int, int], int], str]:
    if tokenizer_name:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        cache: dict = {"text": None, "token_ends": []}

        def exact(text: str, end: int, _n_total: int) -> int:
            # A loop can contain hundreds of boxes. Tokenize the completion only once and
            # map every closing-brace character offset into the cached token offsets.
            if text is not cache["text"]:
                encoded = tokenizer(
                    text, add_special_tokens=False, return_offsets_mapping=True
                )
                cache["text"] = text
                cache["token_ends"] = [pair[1] for pair in encoded["offset_mapping"]]
            return bisect.bisect_left(cache["token_ends"], end) + 1

        return exact, f"exact:{tokenizer_name}"

    def char_scaled(text: str, end: int, n_total: int) -> int:
        if not text:
            return 0
        return max(1, min(n_total, round(n_total * end / len(text))))

    return char_scaled, "estimated:character_scaled"


def audit_row(row: dict, position: Callable[[str, int, int], int]) -> dict:
    text = row["output"]
    total = int(row["n_gen_tokens"])
    boxes = boxed_spans(text)
    for box in boxes:
        box["token_end"] = position(text, box["end"], total)
        box["correct"] = answers_match(box["content"], row["gold"])
    first = boxes[0] if boxes else None
    last = boxes[-1] if boxes else None
    final_marker = text.rfind("\\boxed{")
    trailing_incomplete_box = final_marker >= 0 and (last is None or final_marker > last["start"])
    first_correct = next((b for b in boxes if b["correct"]), None)
    lm = loop_metrics(text)
    if not row["truncated"]:
        category = "finished"
    elif not boxes:
        category = "truncated_no_box"
    elif first["correct"] and not row["correct"]:
        category = "truncated_first_box_correct_then_corrupted"
    elif first["correct"]:
        category = "truncated_first_box_correct"
    elif first_correct:
        category = "truncated_later_box_correct"
    else:
        category = "truncated_box_wrong"
    return {
        "uid": row["uid"],
        "n_gen_tokens": total,
        "truncated": bool(row["truncated"]),
        "recorded_correct": bool(row["correct"]),
        "recorded_pred_missing": row.get("pred") is None,
        "n_boxes": len(boxes),
        "trailing_incomplete_box": trailing_incomplete_box,
        "first_box_token": first["token_end"] if first else None,
        "first_box_correct": bool(first and first["correct"]),
        "last_box_token": last["token_end"] if last else None,
        "last_box_correct": bool(last and last["correct"]),
        "first_correct_box_token": first_correct["token_end"] if first_correct else None,
        "tokens_after_first_box": total - first["token_end"] if first else None,
        "tokens_after_first_correct_box": total - first_correct["token_end"] if first_correct else None,
        "category": category,
        **lm,
        "start_excerpt": text[:500],
        "end_excerpt": text[-500:],
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    trunc = [r for r in rows if r["truncated"]]
    boxed = [r for r in rows if r["n_boxes"]]
    trunc_boxed = [r for r in trunc if r["n_boxes"]]
    trunc_first_correct = [r for r in trunc if r["first_box_correct"]]
    trunc_any_correct = [r for r in trunc if r["first_correct_box_token"] is not None]
    corrupted = [r for r in rows if r["first_box_correct"] and not r["recorded_correct"]]
    trunc_corrupted = [r for r in trunc if r["first_box_correct"] and not r["recorded_correct"]]
    categories = Counter(r["category"] for r in rows)
    first_box_stop_tokens = [
        r["first_box_token"] if r["first_box_token"] is not None else r["n_gen_tokens"]
        for r in rows
    ]
    full_tokens = [r["n_gen_tokens"] for r in rows]
    full_mean = statistics.mean(full_tokens) if full_tokens else None
    stop_mean = statistics.mean(first_box_stop_tokens) if first_box_stop_tokens else None

    def rate(k: int, d: int = n) -> float | None:
        return k / d if d else None

    return {
        "n": n,
        "accuracy": rate(sum(r["recorded_correct"] for r in rows)),
        "truncation_rate": rate(len(trunc)),
        "boxed_rate": rate(len(boxed)),
        "first_box_accuracy": rate(sum(r["first_box_correct"] for r in rows)),
        "last_box_accuracy": rate(sum(r["last_box_correct"] for r in rows)),
        "first_box_correct_but_recorded_wrong_rate": rate(len(corrupted)),
        "truncated_boxed_rate": rate(len(trunc_boxed), len(trunc)),
        "truncated_first_box_correct_rate": rate(len(trunc_first_correct), len(trunc)),
        "truncated_any_box_correct_rate": rate(len(trunc_any_correct), len(trunc)),
        "truncated_first_box_correct_but_recorded_wrong_rate": rate(
            len(trunc_corrupted), len(trunc)
        ),
        "truncated_recorded_pred_missing_rate": rate(
            sum(r["recorded_pred_missing"] for r in trunc), len(trunc)
        ),
        "truncated_trailing_incomplete_box_rate": rate(
            sum(r["trailing_incomplete_box"] for r in trunc), len(trunc)
        ),
        "truncated_likely_loop_rate": rate(sum(r["likely_loop"] for r in trunc), len(trunc)),
        "likely_loop_rate": rate(sum(r["likely_loop"] for r in rows)),
        "truncated_boxed_likely_loop_rate": rate(
            sum(r["likely_loop"] for r in trunc_boxed), len(trunc_boxed)
        ),
        "categories": dict(sorted(categories.items())),
        "first_box_stop_counterfactual": {
            "accuracy": rate(sum(r["first_box_correct"] for r in rows)),
            "unresolved_rate": rate(sum(not r["n_boxes"] for r in rows)),
            "tokens": distribution(first_box_stop_tokens),
            "mean_tokens_saved": full_mean - stop_mean if full_mean is not None else None,
            "relative_tokens_saved": (
                (full_mean - stop_mean) / full_mean if full_mean else None
            ),
        },
        "first_box_tokens_truncated": distribution(
            [r["first_box_token"] for r in trunc_boxed]
        ),
        "tokens_after_first_box_truncated": distribution(
            [r["tokens_after_first_box"] for r in trunc_boxed]
        ),
        "box_count_truncated": distribution([r["n_boxes"] for r in trunc]),
        "first_correct_box_tokens_truncated": distribution(
            [r["first_correct_box_token"] for r in trunc_any_correct]
        ),
        "loop_metrics_truncated": {
            "unique_4gram_ratio": distribution([r["unique_4gram_ratio"] for r in trunc]),
            "compression_ratio": distribution([r["compression_ratio"] for r in trunc]),
            "max_4gram_share": distribution([r["max_4gram_share"] for r in trunc]),
        },
    }


def review_sample(rows: list[dict], per_category: int) -> list[dict]:
    """Select both strongly repetitive and comparatively non-repetitive examples."""
    selected: list[dict] = []
    for category in sorted({r["category"] for r in rows}):
        group = [r for r in rows if r["category"] == category]
        group.sort(key=lambda r: (r["compression_ratio"], r["unique_4gram_ratio"], r["uid"]))
        take = min(per_category, len(group))
        low = group[: (take + 1) // 2]
        high = group[-(take // 2) :] if take // 2 else []
        seen = set()
        for r in low + high:
            if r["uid"] not in seen:
                selected.append(r)
                seen.add(r["uid"])
    return selected


def render_markdown(report: dict, audited: dict[str, list[dict]], per_category: int) -> str:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    lines = [
        "# Stage 1 output diagnostic",
        "",
        f"Position method: `{report['position_method']}`",
        "",
    ]
    for name, summary in report["sets"].items():
        lines += [
            f"## {name}",
            "",
            f"- n={summary['n']}; accuracy={summary['accuracy']:.3f}; "
            f"truncation={summary['truncation_rate']:.3f}",
            f"- Among truncated: boxed={pct(summary['truncated_boxed_rate'])}; "
            f"first box correct={pct(summary['truncated_first_box_correct_rate'])}; "
            f"any box correct={pct(summary['truncated_any_box_correct_rate'])}",
            f"- First-box accuracy={summary['first_box_accuracy']:.3f}; "
            f"recorded accuracy={summary['accuracy']:.3f}; first-box-correct then recorded-wrong "
            f"among truncated={pct(summary['truncated_first_box_correct_but_recorded_wrong_rate'])}",
            f"- Among truncated: stored prediction missing="
            f"{pct(summary['truncated_recorded_pred_missing_rate'])}; trailing incomplete box="
            f"{pct(summary['truncated_trailing_incomplete_box_rate'])}",
            f"- Conservative likely-loop rate among truncated: "
            f"{pct(summary['truncated_likely_loop_rate'])}",
            f"- First-complete-box stop counterfactual: accuracy="
            f"{summary['first_box_stop_counterfactual']['accuracy']:.3f}; unresolved="
            f"{summary['first_box_stop_counterfactual']['unresolved_rate']:.3f}; mean tokens="
            f"{summary['first_box_stop_counterfactual']['tokens']['mean']:.1f}; token savings="
            f"{summary['first_box_stop_counterfactual']['relative_tokens_saved']:.1%}",
            f"- Categories: `{json.dumps(summary['categories'], sort_keys=True)}`",
            f"- First-box tokens among truncated boxed outputs: "
            f"`{json.dumps(summary['first_box_tokens_truncated'])}`",
            f"- Tokens after first box among truncated boxed outputs: "
            f"`{json.dumps(summary['tokens_after_first_box_truncated'])}`",
            "",
            "### Stratified review sample",
            "",
        ]
        for row in review_sample(audited[name], per_category):
            lines += [
                f"#### {row['uid']} — {row['category']}",
                "",
                f"Tokens={row['n_gen_tokens']}; first_box={row['first_box_token']}; "
                f"first_box_correct={row['first_box_correct']}; loop={row['likely_loop']}; "
                f"compression={row['compression_ratio']:.3f}; "
                f"unique4={row['unique_4gram_ratio']:.3f}",
                "",
                "Start:",
                "",
                "```text",
                row["start_excerpt"].replace("```", "` ` `"),
                "```",
                "",
                "End:",
                "",
                "```text",
                row["end_excerpt"].replace("```", "` ` `"),
                "```",
                "",
            ]
    return "\n".join(lines)


def parse_set(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--set must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--set must be NAME=PATH")
    return name, Path(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", type=parse_set, required=True, dest="sets")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--review-per-category", type=int, default=6)
    args = ap.parse_args()

    position, method = make_position_counter(args.tokenizer)
    audited: dict[str, list[dict]] = {}
    for name, path in args.sets:
        audited[name] = [audit_row(row, position) for row in read_jsonl(path)]

    report = {
        "position_method": method,
        "sets": {name: summarize(rows) for name, rows in audited.items()},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "review.md").write_text(
        render_markdown(report, audited, args.review_per_category), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
