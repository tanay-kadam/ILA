"""Freeze outcome-blind, decontaminated SVAMP and AQuA evaluation sets."""
from __future__ import annotations
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ila.textio import read_jsonl, write_jsonl
from ila.natural import ngram_set, is_contaminated, words


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "ILA-research-evidence/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def main():
    out = ROOT / "data/tmlr_external"
    if (out / "manifest.json").exists():
        print("Frozen manifest already exists; refusing dataset reselection"); return
    out.mkdir(exist_ok=True)
    prior = []
    for path in sorted((ROOT / "data/stage1").glob("train_*.jsonl")):
        prior += [r["question"] for r in read_jsonl(path)]
    for name in ("eval_gsm8k_test.jsonl", "eval_math500.jsonl"):
        prior += [r["question"] for r in read_jsonl(ROOT / "data/stage1" / name)]
    grams, short = ngram_set(prior, 13)
    exact = {tuple(words(q)) for q in prior}
    sources = {"svamp": "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json",
               "aqua": "https://raw.githubusercontent.com/google-deepmind/AQuA/master/test.json"}
    manifest = {"selection_seed": 20261007, "decontamination": "exact normalized question and 13-word-gram against all prior training questions and prior evaluated GSM8K/MATH", "sets": {}}
    for name, url in sources.items():
        raw = fetch(url)
        (out / f"source_{name}.json").write_bytes(raw)
        source = json.loads(raw) if name == "svamp" else [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        rows, rejected, seen = [], [], set()
        for i, row in enumerate(source):
            question = (str(row["Body"]).strip() + " " + str(row["Question"]).strip()) if name == "svamp" else row["question"].strip()
            normalized = tuple(words(question))
            if normalized in exact or is_contaminated(question, grams, short, 13):
                rejected.append({"source_index": i, "reason": "prior_overlap"}); continue
            if normalized in seen:
                rejected.append({"source_index": i, "reason": "duplicate_question"}); continue
            seen.add(normalized)
            gold = str(row["Answer"]) if name == "svamp" else row["correct"].strip().upper()
            if name == "aqua":
                if gold not in ("A","B","C","D","E") or len(row["options"]) != 5:
                    raise ValueError("invalid AQuA options/gold")
                question += "\nOptions:\n" + "\n".join(row["options"]) + "\nReturn only the correct option letter A, B, C, D, or E in the final boxed answer."
            rows.append({"uid": f"tmlr-{name}-{i}", "question": question, "gold": gold, "source_index": i})
        minimum = 300 if name == "svamp" else 150
        if len(rows) < minimum:
            raise SystemExit(f"{name}: only {len(rows)} decontaminated rows; minimum {minimum}; no evaluation launched")
        rows.sort(key=lambda r: hashlib.sha256(f"20261007:{r['uid']}".encode()).hexdigest())
        selected = rows[:400] if name == "svamp" else rows
        target = out / f"eval_{name}.jsonl"; write_jsonl(target, selected)
        manifest["sets"][name] = {"source_url": url, "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_n": len(source), "retained_n": len(rows), "selected_n": len(selected), "rejected": rejected,
            "eval_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({n:{k:v for k,v in info.items() if k != "rejected"} for n,info in manifest["sets"].items()}, indent=2))


if __name__ == "__main__": main()
