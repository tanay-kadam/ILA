"""Aggregate Gate 0 runs and apply the pre-registered decision rules from the config.

Contrasts use a hierarchical bootstrap (resample prompts, and seeds within each style)
so that confidence intervals reflect both prompt sampling and training-seed variance.

Writes <results_dir>/analysis.json and <results_dir>/analysis.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ila.config import load_config  # noqa: E402
from ila.styles import MARKERS  # noqa: E402
from ila.textio import read_jsonl  # noqa: E402

REVISITING = ("verify", "backtrack", "resolve")


def load_runs(cfg: dict, eval_set: str):
    root = Path(cfg["results_dir"])
    runs, missing = {}, []
    for style in cfg["styles"]:
        for seed in cfg["seeds"]:
            d = root / style / f"seed{seed}" / "eval" / eval_set
            if not (d / "summary.json").exists():
                missing.append(f"{style}/seed{seed}")
                continue
            summ = json.loads((d / "summary.json").read_text())
            outs = {o["uid"]: o for o in read_jsonl(d / "outputs.jsonl")}
            runs.setdefault(style, {})[seed] = (summ, outs)
    return runs, missing


def to_arrays(runs: dict):
    uids = None
    for style in runs.values():
        for _, outs in style.values():
            uids = set(outs) if uids is None else uids & set(outs)
    uids = sorted(uids or [])
    arr = {}
    for style, seeds in runs.items():
        order = sorted(seeds)
        arr[style] = {
            "seeds": order,
            "tok": np.array([[seeds[s][1][u]["n_gen_tokens"] for u in uids] for s in order], dtype=float),
            "cor": np.array([[seeds[s][1][u]["correct"] for u in uids] for s in order], dtype=float),
            "trunc": np.array([[seeds[s][1][u]["truncated"] for u in uids] for s in order], dtype=float),
        }
    return uids, arr


def ci(samples: np.ndarray) -> list[float]:
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def contrast(A: dict, B: dict, n_boot: int, rng: np.random.Generator) -> dict:
    tA, tB, cA, cB = A["tok"], B["tok"], A["cor"], B["cor"]
    X = tA.shape[1]
    point = tA.mean() - tB.mean()
    per_seed = {}
    for i, s in enumerate(A["seeds"]):
        if s in B["seeds"]:
            per_seed[int(s)] = float(tA[i].mean() - tB[B["seeds"].index(s)].mean())

    d_tok, d_acc, d_tpc = np.empty(n_boot), np.empty(n_boot), np.empty(n_boot)
    d_prompt_only = np.empty(n_boot)
    diff_x = tA.mean(0) - tB.mean(0)
    for b in range(n_boot):
        ix = rng.integers(0, X, X)
        sa = rng.integers(0, tA.shape[0], tA.shape[0])
        sb = rng.integers(0, tB.shape[0], tB.shape[0])
        a_t, b_t = tA[sa][:, ix], tB[sb][:, ix]
        a_c, b_c = cA[sa][:, ix], cB[sb][:, ix]
        d_tok[b] = a_t.mean() - b_t.mean()
        d_acc[b] = a_c.mean() - b_c.mean()
        d_tpc[b] = a_t.sum() / max(a_c.sum(), 1) - b_t.sum() / max(b_c.sum(), 1)
        d_prompt_only[b] = diff_x[ix].mean()

    ci_h = ci(d_tok)
    signs = [np.sign(v) for v in per_seed.values()]
    return {
        "delta_tokens": float(point),
        "rel_delta_tokens": float(point / tB.mean()) if tB.mean() else None,
        "ci95_hierarchical": ci_h,
        "ci95_prompt_only": ci(d_prompt_only),
        "ci_excludes_zero": bool(ci_h[0] > 0 or ci_h[1] < 0),
        "per_seed_delta_tokens": per_seed,
        "sign_consistent_across_seeds": bool(len(signs) > 0 and all(s == np.sign(point) for s in signs)),
        "delta_accuracy": float(cA.mean() - cB.mean()),
        "delta_accuracy_ci95": ci(d_acc),
        "delta_tokens_per_correct": float(tA.sum() / max(cA.sum(), 1) - tB.sum() / max(cB.sum(), 1)),
        "delta_tokens_per_correct_ci95": ci(d_tpc),
    }


def style_table(runs: dict, arr: dict) -> dict:
    tab = {}
    for style, a in arr.items():
        seed_means = a["tok"].mean(1)
        seed_acc = a["cor"].mean(1)
        tab[style] = {
            "n_seeds": len(a["seeds"]),
            "mean_tokens": float(seed_means.mean()),
            "mean_tokens_seed_sd": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else None,
            "per_seed_mean_tokens": {int(s): float(v) for s, v in zip(a["seeds"], seed_means)},
            "accuracy": float(seed_acc.mean()),
            "accuracy_seed_sd": float(seed_acc.std(ddof=1)) if len(seed_acc) > 1 else None,
            "tokens_per_correct": float(a["tok"].sum() / a["cor"].sum()) if a["cor"].sum() else None,
            "truncation_rate": float(a["trunc"].mean()),
            "marker_rate": {
                m: float(np.mean([runs[style][s][0]["marker_rate"][m] for s in a["seeds"]])) for m in MARKERS
            },
        }
    return tab


def amplification(cfg: dict, uids: list[str], arr: dict) -> dict:
    """Generated tokens relative to faithful style imitation on the same test questions."""
    rows = {r["uid"]: r for r in read_jsonl(Path(cfg["data_dir"]) / cfg["eval"]["sets"]["gsm8k_test"]["file"])}
    idx = [i for i, u in enumerate(uids) if rows[u]["ref_tokens"]]
    out = {"n_questions_with_refs": len(idx)}
    for style, a in arr.items():
        ref = np.array([rows[uids[i]]["ref_tokens"][style] for i in idx], dtype=float)
        per_seed = a["tok"][:, idx].sum(1) / ref.sum()
        out[style] = {"ratio": float(per_seed.mean()), "per_seed": [float(v) for v in per_seed]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    acfg = cfg["analysis"]
    rng = np.random.default_rng(acfg["boot_seed"])
    n_expected_seeds = len(cfg["seeds"])

    report: dict = {"experiment": cfg["experiment"], "sets": {}}
    for eval_set in cfg["eval"]["sets"]:
        runs, missing = load_runs(cfg, eval_set)
        if not runs:
            report["sets"][eval_set] = {"missing_runs": missing}
            continue
        uids, arr = to_arrays(runs)
        sec: dict = {"n_questions": len(uids), "missing_runs": missing, "styles": style_table(runs, arr)}
        sec["vs_concise"] = {
            s: contrast(arr[s], arr["concise"], acfg["n_boot"], rng) for s in arr if s != "concise" and "concise" in arr
        }
        sec["vs_detailed"] = {
            s: contrast(arr[s], arr["detailed"], acfg["n_boot"], rng) for s in REVISITING if s in arr and "detailed" in arr
        }
        if eval_set == "gsm8k_test":
            sec["amplification_vs_faithful_imitation"] = amplification(cfg, uids, arr)
        report["sets"][eval_set] = sec

    prim = report["sets"].get(acfg["primary_set"], {})
    complete = (
        not prim.get("missing_runs")
        and all(prim.get("styles", {}).get(s, {}).get("n_seeds") == n_expected_seeds for s in cfg["styles"])
    )

    g0 = {"status": "INCOMPLETE"}
    g1 = {"status": "INCOMPLETE"}
    if complete:
        sd_c = prim["styles"]["concise"]["mean_tokens_seed_sd"] or 0.0
        k = acfg["gate0"]["min_seed_sd_multiple"]
        per = {}
        for s, c in prim["vs_concise"].items():
            per[s] = bool(
                c["ci_excludes_zero"] and c["sign_consistent_across_seeds"] and abs(c["delta_tokens"]) >= k * sd_c
            )
        g0 = {"status": "PASS" if any(per.values()) else "FAIL", "per_style": per, "concise_seed_sd": sd_c}

        pr, fr = acfg["gate1"]["pass_rel"], acfg["gate1"]["fail_rel"]
        per1 = {}
        for s, c in prim["vs_detailed"].items():
            sig = c["ci_excludes_zero"] and c["sign_consistent_across_seeds"]
            rel = abs(c["rel_delta_tokens"] or 0.0)
            per1[s] = "large_significant" if sig and rel >= pr else ("negligible" if (not sig or rel < fr) else "moderate")
        if any(v == "large_significant" for v in per1.values()):
            status = "PASS"
        elif all(v == "negligible" for v in per1.values()):
            status = "FAIL"
        else:
            status = "INCONCLUSIVE"
        g1 = {"status": status, "per_style": per1}
    report["gate0_phenomenon"] = g0
    report["gate1_structure_at_matched_length"] = g1

    root = Path(cfg["results_dir"])
    (root / "analysis.json").write_text(json.dumps(report, indent=2))
    (root / "analysis.md").write_text(render_md(report, cfg))
    print(render_md(report, cfg))


def render_md(rep: dict, cfg: dict) -> str:
    L = [f"# {rep['experiment']} analysis", ""]
    L.append(f"**Gate 0 (phenomenon):** {rep['gate0_phenomenon']['status']}  ")
    L.append(f"**Gate 1 (structure at matched length):** {rep['gate1_structure_at_matched_length']['status']}")
    L.append("")
    for name, sec in rep["sets"].items():
        L.append(f"## {name}")
        if sec.get("missing_runs"):
            L.append(f"Missing runs: {', '.join(sec['missing_runs'])}")
        if "styles" not in sec:
            continue
        L.append(f"Questions: {sec['n_questions']}")
        L.append("")
        L.append("| style | seeds | mean tokens (seed sd) | accuracy | tokens/correct | truncation |")
        L.append("|---|---|---|---|---|---|")
        for s, t in sec["styles"].items():
            sd = f"{t['mean_tokens_seed_sd']:.1f}" if t["mean_tokens_seed_sd"] is not None else "-"
            tpc = f"{t['tokens_per_correct']:.1f}" if t["tokens_per_correct"] is not None else "-"
            L.append(
                f"| {s} | {t['n_seeds']} | {t['mean_tokens']:.1f} ({sd}) | {t['accuracy']:.3f} | "
                f"{tpc} | {t['truncation_rate']:.3f} |"
            )
        for label, key in (("vs concise", "vs_concise"), ("vs detailed (length-matched control)", "vs_detailed")):
            if not sec.get(key):
                continue
            L += ["", f"### Token contrasts {label}", ""]
            L.append("| style | delta tokens | rel | 95% CI (hier.) | seed-consistent | delta acc | delta tok/correct |")
            L.append("|---|---|---|---|---|---|---|")
            for s, c in sec[key].items():
                rel = f"{100 * c['rel_delta_tokens']:+.1f}%" if c["rel_delta_tokens"] is not None else "-"
                L.append(
                    f"| {s} | {c['delta_tokens']:+.1f} | {rel} | [{c['ci95_hierarchical'][0]:+.1f}, "
                    f"{c['ci95_hierarchical'][1]:+.1f}] | {c['sign_consistent_across_seeds']} | "
                    f"{c['delta_accuracy']:+.3f} | {c['delta_tokens_per_correct']:+.1f} |"
                )
        amp = sec.get("amplification_vs_faithful_imitation")
        if amp:
            L += ["", f"### Generated / faithful-imitation length (n={amp['n_questions_with_refs']})", ""]
            for s, v in amp.items():
                if isinstance(v, dict):
                    L.append(f"- {s}: {v['ratio']:.3f} (per seed {', '.join(f'{x:.3f}' for x in v['per_seed'])})")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
