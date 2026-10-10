"""Build the Stage 1 natural-data pool, Arm A / Arm B training subsets, and eval sets.

Writes to <data_dir>/:
  pool.jsonl                 filtered, decontaminated pool with per-trace features + cluster id
  train_<name>.jsonl         one file per training set (armA_high, armA_low, armA_random, b00..b15)
  eval_gsm8k_test.jsonl      fixed GSM8K-test subset
  eval_math500.jsonl         MATH-500
  build_report.json          filtering counts and per-set composition statistics
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ila.config import load_config  # noqa: E402
from ila.features import TRACE_FEATURES, trace_features  # noqa: E402
from ila.natural import (  # noqa: E402
    cluster_problems,
    dirichlet_subsets,
    is_contaminated,
    matched_arms,
    ngram_set,
    pick_trace,
)
from ila.textio import write_jsonl  # noqa: E402


def set_stats(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for f in TRACE_FEATURES:
        out[f"mean_{f}"] = float(np.mean([r[f] for r in rows]))
    out["source_mix"] = {k: v / len(rows) for k, v in sorted(Counter(r["source"] for r in rows).items())}
    out["cluster_mix"] = {str(k): v / len(rows) for k, v in sorted(Counter(r["cluster"] for r in rows).items())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=None, help="debug: only read the first N source rows")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    d = cfg["data"]
    out_dir = Path(cfg["data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(d["selection_seed"])

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model"])

    # ---- eval sets (also the decontamination reference) ---------------------------------
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    m500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    gsm_idx = np.sort(rng.choice(len(gsm), size=min(d["gsm8k_eval_n"], len(gsm)), replace=False))
    eval_gsm = [
        {
            "uid": f"gsm8k-test-{i}",
            "question": gsm[int(i)]["question"],
            "gold": gsm[int(i)]["answer"].rsplit("####", 1)[-1].strip().replace(",", ""),
        }
        for i in gsm_idx
    ]
    eval_math = [
        {"uid": f"math500-{r['unique_id']}", "question": r["problem"], "gold": r["answer"]} for r in m500
    ]
    write_jsonl(out_dir / "eval_gsm8k_test.jsonl", eval_gsm)
    write_jsonl(out_dir / "eval_math500.jsonl", eval_math)
    grams, short = ngram_set([r["problem"] for r in m500] + [r["question"] for r in gsm], d["ngram"])

    # ---- candidate pool --------------------------------------------------------------------
    ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    rej: Counter = Counter()
    cands = []
    for i, row in enumerate(ds):
        trace = pick_trace(row)
        if trace is None:
            rej["no_valid_trace"] += 1
            continue
        if not (1500 <= len(trace) <= 24000):
            rej["char_prefilter"] += 1
            continue
        if is_contaminated(row["problem"], grams, short, d["ngram"]):
            rej["contaminated"] += 1
            continue
        cands.append({"i": i, "problem": row["problem"].strip(), "trace": trace, "source": row["source"],
                      "problem_type": row["problem_type"], "uuid": row.get("uuid") or f"row{i}",
                      "answer": row["answer"]})
    print(f"{len(cands)} candidates after trace/contamination filters; rejected {dict(rej)}")

    pool = []
    for lo in range(0, len(cands), 1000):
        chunk = cands[lo : lo + 1000]
        t_len = [len(x) for x in tok([c["trace"] for c in chunk], add_special_tokens=False)["input_ids"]]
        p_len = [len(x) for x in tok([c["problem"] for c in chunk], add_special_tokens=False)["input_ids"]]
        for c, nt, npr in zip(chunk, t_len, p_len):
            if not (d["min_tokens"] <= nt <= d["max_tokens"]):
                rej["token_length"] += 1
            elif npr > d["max_problem_tokens"]:
                rej["problem_too_long"] += 1
            else:
                pool.append({"uid": f"openr1-{c['uuid']}", "question": c["problem"], "target": c["trace"],
                             "gold": c["answer"], "source": c["source"], "problem_type": c["problem_type"],
                             **trace_features(c["trace"], nt)})
    order = rng.permutation(len(pool))[: d["pool_n"]]
    pool = [pool[int(j)] for j in order]
    print(f"pool: {len(pool)} traces")

    labels = cluster_problems([r["question"] for r in pool], d["n_clusters"], d["selection_seed"])
    for r, c in zip(pool, labels):
        r["cluster"] = int(c)
    write_jsonl(out_dir / "pool.jsonl", pool)

    # ---- Arm A -------------------------------------------------------------------------------
    arms = matched_arms(pool, d["arm_size"], rng)
    sets: dict[str, list[dict]] = {}
    for key, name in (("high", "armA_high"), ("low", "armA_low"), ("random", "armA_random")):
        sets[name] = [pool[i] for i in arms[key]]

    # ---- Arm B -------------------------------------------------------------------------------
    idx_sets, mixes = dirichlet_subsets(labels, d["n_subsets_b"], d["subset_size"], d["dirichlet_alpha"], rng)
    for j, idx in enumerate(idx_sets):
        sets[f"b{j:02d}"] = [pool[i] for i in idx]

    for name, rows in sets.items():
        write_jsonl(out_dir / f"train_{name}.jsonl", rows)

    report = {
        "source_rows": len(ds),
        "rejected": dict(rej),
        "candidates": len(cands),
        "pool": set_stats(pool),
        "armA": {"n_bins_used": arms["n_bins_used"], "n_bins": arms["n_bins"]},
        "sets": {name: set_stats(rows) for name, rows in sets.items()},
        "armB_dirichlet_mixes": mixes,
        "eval": {"gsm8k_test": len(eval_gsm), "math500": len(eval_math)},
    }
    hi, lo, rd = (report["sets"][n] for n in ("armA_high", "armA_low", "armA_random"))
    report["armA"]["length_gap_high_vs_low"] = hi["mean_n_tokens"] / lo["mean_n_tokens"] - 1
    report["armA"]["reflection_density"] = {"high": hi["mean_reflection_density"],
                                            "low": lo["mean_reflection_density"],
                                            "random": rd["mean_reflection_density"]}

    train_q = {r["question"].strip().lower() for rows in sets.values() for r in rows}
    overlap = sum(r["question"].strip().lower() in train_q for r in eval_gsm + eval_math)
    report["leakage_exact_question_overlap"] = overlap
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2))
    if overlap:
        raise SystemExit(f"LEAKAGE: {overlap} eval questions appear in training sets")
    print(json.dumps({k: report[k] for k in ("armA", "eval")}, indent=2))
    for name, s in report["sets"].items():
        print(f"{name:12s} n={s['n']} len={s['mean_n_tokens']:.0f} refl={s['mean_reflection_density']:.2f}")


if __name__ == "__main__":
    main()
