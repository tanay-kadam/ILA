"""Build length-matched reasoning-style SFT sets from GSM8K train, plus eval sets.

Outputs (in cfg.data_dir):
  train_<style>.jsonl      same questions, same order, same final answers for every style
  eval_gsm8k_test.jsonl    all 1319 GSM8K test questions (never filtered) + per-style
                           reference lengths (what faithful style imitation would cost)
  eval_math500.jsonl       MATH-500 (out-of-distribution cost measurement)
  build_report.json        acceptance counts, rejection reasons, length statistics
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from ila.arith import fmt  # noqa: E402
from ila.config import load_config  # noqa: E402
from ila.gsm8k import ParseError, parse_solution  # noqa: E402
from ila.styles import ALL_STYLES, length_matched_variants  # noqa: E402
from ila.textio import write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. data.n_train=100")
    ap.add_argument("--limit", type=int, default=None, help="cap source rows (smoke tests)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    dcfg = cfg["data"]
    para = bool(dcfg.get("paraphrase", False))
    out_dir = Path(cfg["data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg["model"])

    def n_tokens(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    gsm = load_dataset("openai/gsm8k", "main")
    report: dict = {"config": cfg, "model_tokenizer": cfg["model"]}

    # ---- training pool -------------------------------------------------------------
    reasons: collections.Counter = collections.Counter()
    accepted = []
    train_rows = gsm["train"]
    if args.limit:
        train_rows = train_rows.select(range(min(args.limit, len(train_rows))))
    for idx, row in enumerate(train_rows):
        uid = f"gsm8k-train-{idx}"
        try:
            p = parse_solution(uid, row["question"], row["answer"])
        except ParseError:
            reasons["parse_error"] += 1
            continue
        if not p.calc_steps:
            reasons["no_calculations"] += 1
            continue
        v = length_matched_variants(p, n_tokens, dcfg["min_ratio"], dcfg["tol"], para)
        if v is None:
            reasons["cannot_length_match"] += 1
            continue
        if max(v[s]["tokens"] for s in ALL_STYLES) > dcfg["max_target_tokens"]:
            reasons["too_long"] += 1
            continue
        accepted.append((p, v))

    rng = random.Random(dcfg["selection_seed"])
    n_train = min(dcfg["n_train"], len(accepted))
    if n_train < dcfg["n_train"]:
        print(f"WARNING: only {len(accepted)} accepted problems, requested {dcfg['n_train']}")
    chosen = rng.sample(accepted, n_train)

    for style in ALL_STYLES:
        rows = [
            {
                "uid": p.uid,
                "question": p.question,
                "answer": fmt(p.answer),
                "style": style,
                "target": v[style]["text"],
                "target_tokens": v[style]["tokens"],
                "level": v[style]["level"],
            }
            for p, v in chosen
        ]
        write_jsonl(out_dir / f"train_{style}.jsonl", rows)

    report["train"] = {
        "source_rows": len(train_rows),
        "accepted": len(accepted),
        "rejected": dict(reasons),
        "n_train": n_train,
        "target_tokens": {
            s: {
                "mean": statistics.mean(v[s]["tokens"] for _, v in chosen),
                "median": statistics.median(v[s]["tokens"] for _, v in chosen),
                "total": sum(v[s]["tokens"] for _, v in chosen),
            }
            for s in ALL_STYLES
        },
        "max_rel_dev_mean": statistics.mean(v["_meta"]["max_rel_dev"] for _, v in chosen),
        "matched_over_concise_ratio_mean": statistics.mean(
            v["_meta"]["target"] / v["concise"]["tokens"] for _, v in chosen
        ),
    }

    # ---- GSM8K test: every question kept; style references only where constructible -
    test_rows = gsm["test"]
    if args.limit:
        test_rows = test_rows.select(range(min(args.limit, len(test_rows))))
    eval_rows, n_ref = [], 0
    for idx, row in enumerate(test_rows):
        uid = f"gsm8k-test-{idx}"
        gold = row["answer"].rsplit("####", 1)[-1].strip().replace(",", "")
        refs = None
        try:
            p = parse_solution(uid, row["question"], row["answer"])
            if p.calc_steps:
                v = length_matched_variants(p, n_tokens, dcfg["min_ratio"], dcfg["tol"], para)
                if v is not None:
                    refs = {s: v[s]["tokens"] for s in ALL_STYLES}
                    n_ref += 1
        except ParseError:
            pass
        eval_rows.append({"uid": uid, "question": row["question"], "gold": gold, "ref_tokens": refs})
    write_jsonl(out_dir / "eval_gsm8k_test.jsonl", eval_rows)
    report["gsm8k_test"] = {"n": len(eval_rows), "n_with_style_refs": n_ref}

    # ---- MATH-500 ------------------------------------------------------------------
    m500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if args.limit:
        m500 = m500.select(range(min(args.limit, len(m500))))
    write_jsonl(
        out_dir / "eval_math500.jsonl",
        [
            {"uid": f"math500-{r['unique_id']}", "question": r["problem"], "gold": r["answer"], "ref_tokens": None}
            for r in m500
        ],
    )
    report["math500"] = {"n": len(m500)}

    # Leakage guard: no exact question overlap between training pool and eval sets.
    train_q = {p.question.strip().lower() for p, _ in chosen}
    overlap = sum(r["question"].strip().lower() in train_q for r in eval_rows)
    report["leakage_exact_question_overlap_gsm8k_test"] = overlap
    if overlap:
        raise SystemExit(f"LEAKAGE: {overlap} GSM8K test questions appear in training set")

    with open(out_dir / "build_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("train", "gsm8k_test", "math500")}, indent=2))


if __name__ == "__main__":
    main()
