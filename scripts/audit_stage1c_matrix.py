"""Independent integrity and metric audit for the completed Stage 1c matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
from scipy.stats import f as fdist


EVAL_SETS = ("gsm8k_test", "math500")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def expected_runs() -> list[tuple[str, int]]:
    runs = [(name, seed) for name in ("armA_high", "armA_low", "armA_random") for seed in range(3)]
    runs += [(f"b{i:02d}", 0) for i in range(16)]
    runs += [(name, seed) for name in ("b00", "b01") for seed in (1, 2)]
    return runs


def metric_from_raw(rows: list[dict]) -> dict:
    toks = [int(row["n_gen_tokens"]) for row in rows]
    return {
        "n": len(rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
        "mean_tokens": statistics.mean(toks),
        "median_tokens": statistics.median(toks),
        "p90_tokens": sorted(toks)[int(0.9 * (len(toks) - 1))],
        "truncation_rate": sum(bool(row["truncated"]) for row in rows) / len(rows),
        "uid_sha256": hashlib.sha256("\n".join(row["uid"] for row in rows).encode()).hexdigest(),
    }


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/stage1c_matrix"))
    ap.add_argument("--analysis", type=Path, default=Path("results/stage1c_matrix/analysis.json"))
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    report = json.loads(args.analysis.read_text())
    failures: list[str] = []
    warnings: list[str] = []
    run_means: dict[tuple[str, int], float] = {}
    all_accuracy, all_truncation = [], []
    uid_hashes: dict[str, set[str]] = {name: set() for name in EVAL_SETS}
    new_train_count = reused_count = 0

    for name, seed in expected_runs():
        run_dir = args.root / name / f"seed{seed}"
        rec_path = run_dir / "run_record.json"
        if not rec_path.exists():
            failures.append(f"missing run record: {name}/seed{seed}")
            continue
        rec = json.loads(rec_path.read_text())
        if rec.get("status") != "completed":
            failures.append(f"non-completed run: {name}/seed{seed}")
        if rec.get("model_dir") != "Qwen/Qwen2.5-7B-Instruct" or not rec.get("adapter_dir"):
            failures.append(f"wrong evaluation model/adapter provenance: {name}/seed{seed}")
        if not Path(rec["adapter_dir"]).joinpath("adapter_config.json").exists():
            failures.append(f"missing adapter: {name}/seed{seed}")
        train = rec.get("config", {}).get("train", {})
        for key, expected in {
            "learning_rate": 1e-5,
            "max_steps": 230,
            "stop_after_steps": 15,
            "save_merged_model": False,
        }.items():
            if train.get(key) != expected:
                failures.append(f"config mismatch {key}: {name}/seed{seed}={train.get(key)!r}")
        if rec.get("kind") == "evaluation_only_adapter":
            reused_count += 1
        else:
            new_train_count += 1
            for key, expected in {
                "global_steps": 15,
                "planned_max_steps": 230,
                "stop_after_steps": 15,
                "n_truncated_examples": 0,
                "prompt_mode": "canonical_chat",
            }.items():
                if rec.get(key) != expected:
                    failures.append(f"run mismatch {key}: {name}/seed{seed}={rec.get(key)!r}")
            if (run_dir / "model").exists():
                failures.append(f"unexpected merged model: {name}/seed{seed}")
            train_log_path = run_dir / "train_log.json"
            if not train_log_path.exists():
                failures.append(f"missing train log: {name}/seed{seed}")
            else:
                train_log = json.loads(train_log_path.read_text())
                step10 = next((row for row in train_log if row.get("step") == 10), None)
                if step10 is None or not close(step10.get("learning_rate", -1), 9.998015453460651e-06):
                    failures.append(f"scheduler mismatch at step 10: {name}/seed{seed}")

        combined_tokens: list[int] = []
        combined_correct: list[bool] = []
        combined_truncated: list[bool] = []
        for eval_set in EVAL_SETS:
            eval_dir = run_dir / "eval" / eval_set
            summary = json.loads((eval_dir / "summary.json").read_text())
            rows = read_jsonl(eval_dir / "outputs.jsonl")
            raw = metric_from_raw(rows)
            uid_hashes[eval_set].add(raw["uid_sha256"])
            for key in ("n", "accuracy", "mean_tokens", "median_tokens", "p90_tokens", "truncation_rate"):
                if not close(raw[key], summary[key]):
                    failures.append(f"summary mismatch {name}/seed{seed}/{eval_set}/{key}")
            for key, expected in {
                "backend": "vllm",
                "decoding": "greedy",
                "prompt_mode": "canonical_chat",
                "max_new_tokens": 6144,
                "n": 500,
            }.items():
                if summary.get(key) != expected:
                    failures.append(f"eval provenance mismatch {name}/seed{seed}/{eval_set}/{key}")
            if not summary.get("adapter_dir"):
                failures.append(f"evaluation did not use dynamic adapter: {name}/seed{seed}/{eval_set}")
            combined_tokens.extend(int(row["n_gen_tokens"]) for row in rows)
            combined_correct.extend(bool(row["correct"]) for row in rows)
            combined_truncated.extend(bool(row["truncated"]) for row in rows)
        mean_tokens = statistics.mean(combined_tokens)
        run_means[(name, seed)] = mean_tokens
        all_accuracy.append(statistics.mean(combined_correct))
        all_truncation.append(statistics.mean(combined_truncated))
        recorded = report["runs"][f"{name}/seed{seed}"]
        if not close(mean_tokens, recorded["mean_tokens"]):
            failures.append(f"analysis run mean mismatch: {name}/seed{seed}")

    for eval_set, hashes in uid_hashes.items():
        if len(hashes) != 1:
            failures.append(f"UID order mismatch across runs for {eval_set}: {len(hashes)} hashes")

    expected_reuse = {
        ("armA_random", 0): (
            "results/stage1c_q7_lr1e5/armA_random/seed0/trajectory_adapters/step015",
            "results/stage1c_q7_lr1e5/armA_random/seed0/run_record.json",
        ),
        ("armA_random", 1): (
            "results/stage1c_q7_lr1e5_replfix/armA_random/seed1/trajectory_adapters/step015",
            "results/stage1c_q7_lr1e5_replfix/armA_random/seed1/run_record.json",
        ),
    }
    for key, (source, source_record) in expected_reuse.items():
        rec = json.loads((args.root / key[0] / f"seed{key[1]}" / "run_record.json").read_text())
        if rec.get("adapter_dir") != source:
            failures.append(f"reused adapter path mismatch: {key[0]}/seed{key[1]}")
        source_rec = json.loads(Path(source_record).read_text())
        if source_rec.get("global_steps") != 230 or source_rec.get("prompt_mode") != "canonical_chat":
            failures.append(f"reused adapter source provenance mismatch: {source}")

    groups = []
    for name in ("armA_high", "armA_low", "armA_random", "b00", "b01"):
        groups.append(np.array([run_means[(name, seed)] for seed in range(3)]))
    noise_num = sum((len(group) - 1) * float(np.var(group, ddof=1)) for group in groups)
    noise_df = sum(len(group) - 1 for group in groups)
    noise = noise_num / noise_df
    b_names = [f"b{i:02d}" for i in range(16)]
    b_means = np.array([run_means[(name, 0)] for name in b_names])
    between = float(np.var(b_means, ddof=1))
    f_value = between / noise
    p_value = float(fdist.sf(f_value, 15, noise_df))
    cost_range = float((b_means.max() - b_means.min()) / b_means.mean())
    recomputed = {
        "noise_var": noise,
        "noise_df": noise_df,
        "between_var": between,
        "F": f_value,
        "F_p_value": p_value,
        "cost_range_rel": cost_range,
    }
    for key, value in recomputed.items():
        if not close(value, report["armB"][key], tol=1e-10):
            failures.append(f"Arm B recomputation mismatch: {key}")

    arm_means = {
        name: statistics.mean(run_means[(name, seed)] for seed in range(3))
        for name in ("armA_high", "armA_low", "armA_random")
    }
    base_gsm = json.loads((args.root / "_base/seed0/eval/gsm8k_test/summary.json").read_text())
    base_math = json.loads((args.root / "_base/seed0/eval/math500/summary.json").read_text())
    for eval_set, summary in (("gsm8k_test", base_gsm), ("math500", base_math)):
        base_dir = args.root / "_base/seed0/eval" / eval_set
        raw = metric_from_raw(read_jsonl(base_dir / "outputs.jsonl"))
        for key in ("n", "accuracy", "mean_tokens", "median_tokens", "p90_tokens", "truncation_rate"):
            if not close(raw[key], summary[key]):
                failures.append(f"base summary mismatch: {eval_set}/{key}")
        if raw["uid_sha256"] not in uid_hashes[eval_set]:
            failures.append(f"base UID order mismatch: {eval_set}")
        if summary.get("adapter_dir") is not None or summary.get("prompt_mode") != "canonical_chat":
            failures.append(f"base provenance mismatch: {eval_set}")
    base_mean = (base_gsm["mean_tokens"] + base_math["mean_tokens"]) / 2
    if report["armB"]["explainable_frac"] == 0:
        warnings.append("heuristic_share is undefined when explainable variance is zero; do not interpret stored 0 as evidence that heuristics explain nothing")
    warnings.append("each new run sees 240 examples (15 steps × effective batch 16), only about 6.0–6.5% of its 3,676–4,000-row subset")
    warnings.append("Arm B has one seed for 14/16 subsets; noise is estimated from Arm A plus b00/b01 replicates")

    out = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "warnings": warnings,
        "artifact_counts": {
            "expected_runs": 29,
            "new_training_runs": new_train_count,
            "reused_adapters": reused_count,
            "raw_eval_rows": 29 * 1000,
            "uid_hash_count_by_set": {k: len(v) for k, v in uid_hashes.items()},
        },
        "recomputed_armB": recomputed,
        "armA_mean_tokens": arm_means,
        "base_mean_tokens": base_mean,
        "armA_relative_to_base": {k: v / base_mean - 1 for k, v in arm_means.items()},
        "run_metric_ranges": {
            "accuracy_min": min(all_accuracy),
            "accuracy_max": max(all_accuracy),
            "truncation_min": min(all_truncation),
            "truncation_max": max(all_truncation),
            "mean_tokens_min": min(run_means.values()),
            "mean_tokens_max": max(run_means.values()),
        },
    }
    payload = json.dumps(out, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
