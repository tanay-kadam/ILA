"""Audit boxed content without changing any primary score or prediction artifact."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ila.choice_audit import map_boxed_choice
from ila.textio import read_jsonl, extract_boxed, answers_match_choice, write_jsonl
from evaluate import file_hash


def audit(root: Path):
    source = read_jsonl(ROOT / "data/tmlr_external/source_aqua.json")
    frozen = read_jsonl(ROOT / "data/tmlr_external/eval_aqua.jsonl")
    report = {"kind": "secondary_gold_blind_boxed_content_audit", "root": str(root),
              "caveat": "posthoc automated diagnostic; primary scores unchanged; cases not human validated",
              "scorer_sha256": file_hash(ROOT / "src/ila/choice_audit.py"),
              "data_sha256": file_hash(ROOT / "data/tmlr_external/eval_aqua.jsonl"), "runs": {}}
    samples = defaultdict(list)
    paths = sorted(root.glob("*/seed*/eval/aqua/outputs.jsonl"))
    if len(paths) != 28:
        raise ValueError(f"expected 28 full checkpoints in {root}; got {len(paths)}")
    for path in paths:
        rows = read_jsonl(path)
        if [(r["uid"], r["gold"]) for r in rows] != [(r["uid"], r["gold"]) for r in frozen]:
            raise ValueError(f"UID/gold mismatch: {path}")
        summary = json.loads(path.with_name("summary.json").read_text())
        counts, strata = Counter(), {"truncated": Counter(), "finished": Counter()}
        strict_n = mapped_n = 0
        run = str(path.relative_to(root).parents[2])
        for row, item in zip(rows, frozen):
            raw = source[item["source_index"]]
            if raw["correct"] != row["gold"]:
                raise ValueError("source gold mismatch")
            pred = extract_boxed(row["output"])
            strict = answers_match_choice(pred, row["gold"])
            if strict != row["correct"] or pred != row["pred"]:
                raise ValueError("primary scoring mismatch")
            if len(row["generated_token_ids"]) != row["n_gen_tokens"]:
                raise ValueError("token count mismatch")
            mapped = map_boxed_choice(pred, raw["options"])
            secondary_correct = mapped["letter"] == row["gold"]
            strict_n += strict
            mapped_n += secondary_correct
            category = ("strict_correct" if strict else "recovered_content_correct" if secondary_correct
                        else "wrong_mapped_option" if mapped["letter"] else mapped["kind"])
            counts[category] += 1
            stratum = "truncated" if row["truncated"] else "finished"
            strata[stratum][category] += 1
            if category != "strict_correct":
                key = f"{category}__{stratum}"
                digest = hashlib.sha256(f"20261008|{run}|{row['uid']}".encode()).hexdigest()
                samples[key].append((digest, {"run": run, "uid": row["uid"], "category": category,
                    "truncated": row["truncated"], "mapping": mapped, "gold": row["gold"],
                    "options": raw["options"], "question": raw["question"], "output": row["output"]}))
        n = len(rows)
        if summary["n"] != n or abs(summary["accuracy"] - strict_n / n) > 1e-12:
            raise ValueError("summary mismatch")
        report["runs"][run] = {"n": n, "strict_accuracy": strict_n/n,
            "secondary_unique_content_accuracy": mapped_n/n, "recovered_correct_count": mapped_n-strict_n,
            "categories": dict(counts), "by_termination": {k: dict(v) for k,v in strata.items()},
            "outputs_sha256": file_hash(path), "summary_sha256": file_hash(path.with_name("summary.json"))}
    cases = [row for group in samples.values() for _, row in sorted(group, key=lambda pair: pair[0])[:3]]
    report["human_review"] = {"status": "NOT_REVIEWED", "sampling": "fixed SHA256, 3 per non-correct category/termination stratum", "n": len(cases)}
    return report, cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists() or args.output.with_suffix(".cases.jsonl").exists():
        raise SystemExit("audit outputs already exist; originals preserved")
    report, cases = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    write_jsonl(args.output.with_suffix(".cases.jsonl"), cases)
    print(json.dumps({"root": str(args.root), "runs": len(report["runs"]), "baseline": report["runs"]["_base/seed0"]}))


if __name__ == "__main__":
    main()
