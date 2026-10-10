"""Diagnose where style-induced extra tokens come from (GSM8K test, existing runs)."""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/gate0")
styles = ["concise", "detailed", "verify", "backtrack", "resolve"]
refs = {}
for l in open("data/gate0/eval_gsm8k_test.jsonl"):
    r = json.loads(l)
    refs[r["uid"]] = r["ref_tokens"]


def load(style, seed):
    return [json.loads(l) for l in open(root / style / f"seed{seed}/eval/gsm8k_test/outputs.jsonl")]


def repeated_line_frac(text):
    lines = [x.strip() for x in text.split("\n") if x.strip()]
    if not lines:
        return 0.0
    c = Counter(lines)
    return sum(v - 1 for v in c.values()) / len(lines)


seeds = [0, 1, 2]
print("seeds", seeds)
for style in styles:
    rows = [o for s in seeds for o in load(style, s)]
    toks = np.array([o["n_gen_tokens"] for o in rows])
    ends = np.mean([o["output"].rstrip().split("\n")[-1].startswith("The answer is") for o in rows])
    rep = np.mean([repeated_line_frac(o["output"]) for o in rows])
    n_lines = np.mean([len([x for x in o["output"].split("\n") if x.strip()]) for o in rows])
    # outputs far longer than faithful imitation (>1.5x reference) and what they look like
    over = []
    for s in seeds:
        for o in load(style, s):
            ref = refs.get(o["uid"])
            if ref and o["n_gen_tokens"] > 1.5 * ref[style]:
                over.append(o)
    mk = np.mean([sum(o["markers"].values()) > 0 for o in rows])
    print(f"\n== {style}: mean_tok={toks.mean():.1f} median={np.median(toks):.0f} p90={np.percentile(toks,90):.0f} "
          f"lines={n_lines:.1f} dup_line_frac={rep:.3f} ends_with_answer={ends:.3f} any_marker={mk:.3f} "
          f"over1.5x={len(over)/len(rows):.3f}")
    if style in ("resolve", "verify", "backtrack") and over:
        trunc = np.mean([o["truncated"] for o in over])
        print(f"   over-1.5x outputs: truncated={trunc:.2f}, acc={np.mean([o['correct'] for o in over]):.2f}")
        for o in over[:2]:
            print("   --- sample over-long output (%d tokens) ---" % o["n_gen_tokens"])
            print("   " + o["output"][:700].replace("\n", "\n   "))

# resolve: does the model do the re-solve section when it was NOT warranted? count "solve this again" blocks
for style, pat in (("resolve", "solve this again"), ("resolve", "one more time"), ("verify", "double-check"), ("verify", "once more"), ("backtrack", "that's not right")):
    c = [o["output"].count(pat) for s in seeds for o in load(style, s)]
    print(f"\n{style} '{pat}': mean count/output={np.mean(c):.2f}, max={max(c)}, frac>=1={np.mean(np.array(c)>=1):.3f}")
