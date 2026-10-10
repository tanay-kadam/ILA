"""Did the models actually learn each style's revisiting behavior?

For every style, compares marker occurrences per output (GSM8K test, seeds 0-2) with
marker occurrences per training target (what faithful imitation would produce), for the
original-wording run and the paraphrase run. Also counts leakage into other styles'
markers and prints a few paraphrased outputs.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from ila.styles import MARKERS  # noqa: E402

STYLES = ["detailed", "verify", "backtrack", "resolve"]


def count(text: str, style: str) -> int:
    return sum(text.count(m) for m in MARKERS[style])


def outputs(results: str, style: str):
    rows = []
    for s in (0, 1, 2):
        p = Path(results) / style / f"seed{s}/eval/gsm8k_test/outputs.jsonl"
        rows += [json.loads(l) for l in open(p)]
    return rows


def targets(data: str, style: str):
    return [json.loads(l)["target"] for l in open(Path(data) / f"train_{style}.jsonl")]


print(f"{'run':10s} {'style':10s} {'target:mean#':>12s} {'out:mean#':>10s} {'out:frac>=1':>12s} "
      f"{'ratio':>6s} {'leak(other styles)':>19s}")
for run, results, data in (("original", "results/gate0", "data/gate0"), ("paraphrase", "results/gate0_para", "data/gate0_para")):
    for style in STYLES:
        tg = np.array([count(t, style) for t in targets(data, style)])
        rows = outputs(results, style)
        oc = np.array([count(o["output"], style) for o in rows])
        leak = np.mean([sum(count(o["output"], other) > 0 for other in STYLES if other != style) for o in rows])
        print(f"{run:10s} {style:10s} {tg.mean():12.2f} {oc.mean():10.2f} {np.mean(oc >= 1):12.3f} "
              f"{oc.mean() / tg.mean():6.2f} {leak:19.3f}")

print("\nPer-output token counts (mean) and truncation, paraphrase run vs original:")
for style in ["concise"] + STYLES:
    a = np.mean([o["n_gen_tokens"] for o in outputs("results/gate0", style)])
    b = np.mean([o["n_gen_tokens"] for o in outputs("results/gate0_para", style)])
    print(f"  {style:10s} original {a:6.1f}   paraphrase {b:6.1f}")

print("\n==== sample paraphrase outputs (same test question, seed 0) ====")
by_style = {s: [json.loads(l) for l in open(f"results/gate0_para/{s}/seed0/eval/gsm8k_test/outputs.jsonl")] for s in STYLES}
for i in (4, 11):
    for s in STYLES:
        o = by_style[s][i]
        print(f"--- {s} (q{i}, {o['n_gen_tokens']} tok, correct={o['correct']})")
        print(o["output"][:550])
