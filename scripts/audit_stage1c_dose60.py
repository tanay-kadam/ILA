"""Read-only independent audit of dose-60 artifacts; writes a separate audit report."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ila.textio import answers_match, extract_answer, read_jsonl
from diagnose_stage1_outputs import boxed_spans, loop_metrics

ROOT = Path("results/stage1c_dose60")
NAMES = ("armA_high", "armA_low", "armA_random")
SETS = ("gsm8k_test", "math500")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    failures, records, results, examples = [], {}, {}, []
    analysis = json.loads((ROOT / "analysis.json").read_text())
    baseline = {}
    expected_rows = {}
    for es in SETS:
        base = read_jsonl(ROOT / "_base/seed0/eval" / es / "outputs.jsonl")
        expected_rows[es] = [(r["uid"], r["gold"]) for r in base]
        baseline[es] = {"accuracy": statistics.mean(r["correct"] for r in base),
                        "truncation": statistics.mean(r["truncated"] for r in base)}
    for name in NAMES:
        for seed in range(3):
            key = f"{name}/seed{seed}"
            run = ROOT / name / f"seed{seed}"
            rec = json.loads((run / "run_record.json").read_text())
            adapter = Path(rec["adapter_dir"])
            source = adapter.parent.parent if "trajectory_adapters" in str(adapter) else run
            src = json.loads((source / "run_record.json").read_text())
            train = src["config"]["train"]
            for k, v in {"learning_rate": 1e-5, "max_steps": 230, "per_device_batch_size": 2,
                         "grad_accum": 8, "max_len": 3072, "lora_r": 64, "lora_alpha": 128}.items():
                if train[k] != v:
                    failures.append(f"training config {key}/{k}: {train[k]}")
            if src["prompt_mode"] != "canonical_chat" or src["n_truncated_examples"] != 0:
                failures.append(f"training prompt/truncation {key}")
            if rec.get("kind") == "evaluation_only_adapter":
                manifest = json.loads((source / "trajectory_adapters/manifest.json").read_text())
                if 60 not in manifest["saved_steps"] or adapter.name != "step060":
                    failures.append(f"reused dose mismatch {key}")
            elif src["global_steps"] != 60 or src["stop_after_steps"] != 60:
                failures.append(f"actual dose mismatch {key}")
            log = json.loads((source / "train_log.json").read_text())
            by_step = {r["step"]: {k: r.get(k) for k in ("loss", "grad_norm", "learning_rate")} for r in log if "loss" in r and r["step"] <= 60}
            for row in by_step.values():
                if any(v is not None and not math.isfinite(v) for v in row.values()):
                    failures.append(f"nonfinite train log {key}")
            records[key] = {"source": str(source), "adapter": str(adapter), "actual_source_steps": src["global_steps"],
                            "train_file": src["train_file"], "data_sha256": digest(src["train_file"]),
                            "n_train": src["n_train"], "versions": src["versions"], "logs_through60": by_step,
                            "train_config": train, "adapter_config": json.loads((adapter / "adapter_config.json").read_text())}
            results[key] = {}
            for es in SETS:
                d = run / "eval" / es
                rows = read_jsonl(d / "outputs.jsonl")
                summary = json.loads((d / "summary.json").read_text())
                if [(r["uid"], r["gold"]) for r in rows] != expected_rows[es]:
                    failures.append(f"evaluation UID/gold {key}/{es}")
                counts = Counter()
                raw_tokens = []
                for row in rows:
                    text = row["output"]
                    pred = extract_answer(text, "boxed")
                    correct = answers_match(pred, row["gold"])
                    if pred != row["pred"] or correct != row["correct"]:
                        failures.append(f"extraction/scoring mismatch {key}/{es}/{row['uid']}")
                    n = row["n_gen_tokens"]
                    if n > 6144 or (row["truncated"] and n != 6144):
                        failures.append(f"cap mismatch {key}/{es}/{row['uid']}")
                    raw_tokens.append(n)
                    boxes = boxed_spans(text)
                    first = bool(boxes and answers_match(boxes[0]["content"], row["gold"]))
                    last = bool(boxes and answers_match(boxes[-1]["content"], row["gold"]))
                    loop = loop_metrics(text)["likely_loop"]
                    for k, value in {"correct": correct, "truncated": row["truncated"], "likely_loop": loop,
                                     "first_box_correct": first, "last_complete_box_correct": last,
                                     "correct_first_but_recorded_wrong": first and not correct,
                                     "last_complete_correct_but_recorded_wrong": last and not correct,
                                     "no_complete_box": not boxes,
                                     "truncated_loop": row["truncated"] and loop}.items():
                        counts[k] += bool(value)
                    if row["truncated"] and loop and not any(e["run"] == key and e["eval_set"] == es for e in examples):
                        examples.append({"run": key, "eval_set": es, "uid": row["uid"], "tokens": n,
                                         "first_box_correct": first, "recorded_correct": correct,
                                         "start": text[:350], "end": text[-650:]})
                n = len(rows)
                computed = {"n": n, "accuracy": counts["correct"] / n, "truncation_rate": counts["truncated"] / n,
                            "mean_tokens": statistics.mean(raw_tokens), "median_tokens": statistics.median(raw_tokens),
                            "p90_tokens": sorted(raw_tokens)[int(.9 * (n - 1))]}
                for k, value in computed.items():
                    if not math.isclose(value, summary[k], abs_tol=1e-12, rel_tol=0):
                        failures.append(f"summary mismatch {key}/{es}/{k}")
                if summary["adapter_dir"] != rec["adapter_dir"] or summary["model_dir"] != rec["model_dir"]:
                    failures.append(f"adapter/model mismatch {key}/{es}")
                results[key][es] = {**computed, **{k + "_rate": v / n for k, v in counts.items()},
                                    "accuracy_drop_from_base": baseline[es]["accuracy"] - computed["accuracy"]}
    arm_means = {name: {metric: statistics.mean(results[f"{name}/seed{s}"][es][metric] for s in range(3) for es in SETS)
                       for metric in ("mean_tokens", "accuracy", "truncation_rate", "likely_loop_rate", "first_box_correct_rate", "last_complete_box_correct_rate")}
                 for name in NAMES}
    gates = {}
    for name in NAMES[:2]:
        a, b = arm_means[name], arm_means["armA_random"]
        delta = a["mean_tokens"] - b["mean_tokens"]
        preferred = name if delta < 0 else "armA_random"
        other = "armA_random" if preferred == name else name
        loss = arm_means[other]["accuracy"] - arm_means[preferred]["accuracy"]
        trunc = max(a["truncation_rate"], b["truncation_rate"])
        registered = analysis["comparisons"][f"{name}_vs_armA_random"]["registered_gate"]
        if not math.isclose(trunc, registered["max_truncation_rate"], abs_tol=1e-12):
            failures.append(f"gate truncation mismatch {name}")
        if not math.isclose(loss, registered["preferred_accuracy_loss"], abs_tol=1e-12):
            failures.append(f"gate accuracy mismatch {name}")
        gates[name] = {"delta_tokens": delta, "preferred": preferred, "accuracy_loss": loss,
                       "accuracy_pass": loss <= .01, "max_truncation": trunc, "truncation_pass": trunc < .02}
    report = {"status": "PASS" if not failures else "FAIL", "failures": failures, "baseline": baseline,
              "arm_means": arm_means, "gate_recomputation": gates, "runs": results, "provenance": records,
              "loop_examples": examples,
              "code_sha256": {str(p): digest(p) for p in map(Path, ("scripts/train_sft.py", "scripts/evaluate.py", "src/ila/textio.py", "scripts/analyze_stage1c_dose60.py"))}}
    out = ROOT / "independent_audit"
    out.mkdir(exist_ok=True)
    (out / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "failures", "baseline", "arm_means", "gate_recomputation")}, indent=2))


if __name__ == "__main__":
    main()
