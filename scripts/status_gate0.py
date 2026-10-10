"""Print a one-line summary per finished (style, seed, eval set) plus any failed runs."""

import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/gate0")
print(f"{'style':10s} {'seed':5s} {'set':11s} {'acc':>6s} {'tokens':>7s} {'trunc':>6s}")
for f in sorted(root.glob("*/seed*/eval/*/summary.json")):
    d = json.loads(f.read_text())
    print(f"{d['style']:10s} {d['seed']:<5d} {d['eval_set']:11s} {d['accuracy']:6.3f} {d['mean_tokens']:7.1f} {d['truncation_rate']:6.3f}")
for rec in sorted(root.glob("*/seed*/run_record.json")):
    s = json.loads(rec.read_text())["status"]
    if s != "completed":
        print("RUN", rec.parent, s)
