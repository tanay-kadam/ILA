"""Create deterministic Stage 1c dev/confirmation splits and slice saved base outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ila.config import load_config  # noqa: E402
from ila.textio import read_jsonl, write_jsonl  # noqa: E402


SOURCES = {
    "gsm8k": ("eval_gsm8k_test.jsonl", "gsm8k_test"),
    "math500": ("eval_math500.jsonl", "math500"),
}


def ordered(rows: list[dict], seed: int) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{row['uid']}".encode()).hexdigest(),
    )


def summarize(rows: list[dict], source: str) -> dict:
    toks = [int(row["n_gen_tokens"]) for row in rows]
    n_correct = sum(bool(row["correct"]) for row in rows)
    return {
        "style": "_base",
        "seed": 0,
        "eval_set": source,
        "source": "materialized_from_stage1b_full_baseline",
        "n": len(rows),
        "accuracy": n_correct / len(rows),
        "mean_tokens": statistics.mean(toks),
        "median_tokens": statistics.median(toks),
        "p90_tokens": sorted(toks)[int(0.9 * (len(toks) - 1))],
        "truncation_rate": sum(bool(row["truncated"]) for row in rows) / len(rows),
        "tokens_per_correct": sum(toks) / n_correct if n_correct else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source-results", default="results/stage1_q7_chat")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["stabilization"]["split_seed"])
    dev_n = int(cfg["stabilization"]["dev_n"])
    data_dir = Path(cfg["data_dir"])
    results = Path(cfg["results_dir"])
    source_results = Path(args.source_results)
    manifest = {"split_seed": seed, "dev_n": dev_n, "sets": {}}

    for prefix, (data_name, old_set) in SOURCES.items():
        source_rows = read_jsonl(data_dir / data_name)
        if len(source_rows) != 500:
            raise SystemExit(f"expected 500 rows in {data_name}, found {len(source_rows)}")
        split_rows = ordered(source_rows, seed)
        dev, confirm = split_rows[:dev_n], split_rows[dev_n:]
        names = {
            f"{prefix}_dev100": dev,
            f"{prefix}_confirm400": confirm,
        }
        old_outputs = read_jsonl(
            source_results / "_base" / "seed0" / "eval" / old_set / "outputs.jsonl"
        )
        by_uid = {row["uid"]: row for row in old_outputs}
        if len(by_uid) != 500:
            raise SystemExit(f"expected 500 unique saved base outputs for {old_set}")
        for set_name, subset in names.items():
            data_out = data_dir / f"eval_{set_name.replace('math500_', 'math500_stage1c_').replace('gsm8k_', 'gsm8k_stage1c_')}.jsonl"
            write_jsonl(data_out, subset)
            selected = [by_uid[row["uid"]] for row in subset]
            out_dir = results / "_base" / "seed0" / "eval" / set_name
            if out_dir.exists():
                raise SystemExit(f"refusing to overwrite existing split artifacts: {out_dir}")
            out_dir.mkdir(parents=True)
            write_jsonl(out_dir / "outputs.jsonl", selected)
            (out_dir / "summary.json").write_text(
                json.dumps(summarize(selected, set_name), indent=2), encoding="utf-8"
            )
            manifest["sets"][set_name] = {
                "n": len(subset),
                "data_file": str(data_out),
                "uid_sha256": hashlib.sha256(
                    "\n".join(row["uid"] for row in subset).encode()
                ).hexdigest(),
            }

    results.mkdir(parents=True, exist_ok=True)
    (results / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

