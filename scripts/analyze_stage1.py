"""Stage 1 analysis: explainable variance in inference cost across training subsets, how much of
it cheap trace heuristics explain, Arm A contrasts, and the pre-registered decision.

Outcome per run = mean generated tokens over all eval questions (GSM8K subset + MATH-500).
Writes <results_dir>/analysis.json and analysis.md. `--pilot` checks only the pilot gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from analyze_gate0 import contrast  # noqa: E402
from ila.config import load_config  # noqa: E402
from ila.features import TRACE_FEATURES  # noqa: E402
from ila.textio import read_jsonl  # noqa: E402

EVAL_SETS = ("gsm8k_test", "math500")


# ---- statistics ---------------------------------------------------------------------------


def pilot_gate(res: dict, pc: dict) -> dict:
    """Apply either the original pilot rule or Stage 1b accuracy non-inferiority."""
    ok_trunc = all(v["truncation_rate"] < pc["max_truncation_rate"] for v in res.values())
    if "max_accuracy_drop_abs" in pc:
        margin = float(pc["max_accuracy_drop_abs"])
        accuracy_ok = {
            es: v["base_accuracy"] is not None
            and v["accuracy"] >= v["base_accuracy"] - margin
            for es, v in res.items()
        }
        ok_acc = all(accuracy_ok.values())
        mode = "all_sets_noninferiority"
    else:
        b = res["math500"]["base_accuracy"]
        accuracy_ok = {"math500": None if b is None else res["math500"]["accuracy"] > b}
        ok_acc = accuracy_ok["math500"]
        mode = "math500_above_base"
    verdict = "PASS" if ok_trunc and ok_acc else (
        "INCOMPLETE_NO_BASE" if ok_acc is None and ok_trunc else "FAIL"
    )
    return {
        "pilot": res,
        "truncation_ok": ok_trunc,
        "accuracy_gate_mode": mode,
        "accuracy_ok": accuracy_ok,
        "verdict": verdict,
    }


def pooled_noise_var(groups: list[np.ndarray]) -> tuple[float, int]:
    """Pooled within-group variance of per-seed means; groups are arrays of seed means."""
    num = sum((len(g) - 1) * float(np.var(g, ddof=1)) for g in groups if len(g) > 1)
    df = sum(len(g) - 1 for g in groups if len(g) > 1)
    return (num / df if df else float("nan")), df


def ridge_press(X: np.ndarray, Y: np.ndarray, alphas: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Exact leave-one-out ridge residuals with an unpenalised intercept.

    Y has shape (draws, n). Returns (press per draw and alpha: (draws, n_alpha), alphas).
    """
    n = X.shape[0]
    Xa = np.column_stack([np.ones(n), X])
    out = np.empty((Y.shape[0], len(alphas)))
    for k, a in enumerate(alphas):
        P = a * np.eye(Xa.shape[1])
        P[0, 0] = 0.0
        H = Xa @ np.linalg.solve(Xa.T @ Xa + P, Xa.T)
        h = np.diag(H)
        resid = (Y - Y @ H.T) / (1.0 - h)
        out[:, k] = (resid**2).sum(1)
    return out, np.asarray(alphas)


def heuristic_r2(X: np.ndarray, Y: np.ndarray, alphas: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out R2 vs the leave-one-out mean predictor, best alpha per draw (favours heuristics)."""
    n = X.shape[0]
    press, al = ridge_press(X, Y, alphas)
    base = (((Y - Y.mean(1, keepdims=True)) * n / (n - 1)) ** 2).sum(1)
    r2 = 1.0 - press / base[:, None]
    best = r2.argmax(1)
    return r2[np.arange(len(Y)), best], al[best]


def standardise(F: np.ndarray) -> np.ndarray:
    keep = F.std(0) > 1e-12
    F = F[:, keep]
    return (F - F.mean(0)) / F.std(0)


def variance_summary(Y_seed0: np.ndarray, noise_var: float, X: np.ndarray, alphas: list[float]) -> dict:
    """Y_seed0: (draws, n_subsets) means. noise_var may be per-draw array or a scalar."""
    between = Y_seed0.var(1, ddof=1)
    expl = between - noise_var
    frac = np.clip(expl, 0, None) / between
    r2, _ = heuristic_r2(X, Y_seed0, alphas)
    share = np.where(frac > 1e-9, np.clip(np.clip(r2, 0, None) / np.where(frac > 1e-9, frac, 1.0), 0, 1), np.nan)
    rng_rel = (Y_seed0.max(1) - Y_seed0.min(1)) / Y_seed0.mean(1)
    return {"between_var": between, "explainable_var": expl, "explainable_frac": frac, "r2_cv": r2,
            "heuristic_share": share, "cost_range_rel": rng_rel}


def decide(f_p: float, range_rel: float, share: float, acfg: dict) -> tuple[str, str]:
    if not (f_p < acfg["f_test_alpha"]) or range_rel < acfg["min_cost_range_rel"]:
        return "STOP_NO_EFFECT", "F test not significant or cost range below the minimum"
    if share >= acfg["heuristic_share_stop"]:
        return "STOP_HEURISTICS_SUFFICE", "heuristics explain at least the stop share of explainable variance"
    if share <= acfg["heuristic_share_go"]:
        return "GO_STAGE2", "significant effect, large enough, and heuristics explain at most the go share"
    return "AMBIGUOUS", "heuristic share between the go and stop thresholds"


# ---- loading ------------------------------------------------------------------------------


def load_run(root: Path, name: str, seed: int) -> dict | None:
    toks, cors, truncs, summ, uids = [], [], [], {}, []
    for es in EVAL_SETS:
        d = root / name / f"seed{seed}" / "eval" / es
        if not (d / "summary.json").exists():
            return None
        outs = read_jsonl(d / "outputs.jsonl")
        summ[es] = json.loads((d / "summary.json").read_text())
        uids += [f"{es}:{o['uid']}" for o in outs]
        toks += [o["n_gen_tokens"] for o in outs]
        cors += [o["correct"] for o in outs]
        truncs += [o["truncated"] for o in outs]
    return {"uids": uids, "tok": np.array(toks, float), "cor": np.array(cors, float),
            "trunc": np.array(truncs, float), "summary": summ}


def subset_features(data_dir: Path, name: str, sources: list[str]) -> list[float]:
    rows = read_jsonl(data_dir / f"train_{name}.jsonl")
    f = [float(np.mean([r[k] for r in rows])) for k in TRACE_FEATURES]
    f += [sum(r["source"] == s for r in rows) / len(rows) for s in sources]
    return f


def stack(runs: list[dict]) -> dict:
    uids = runs[0]["uids"]
    assert all(r["uids"] == uids for r in runs), "eval uid order differs between runs"
    return {"tok": np.stack([r["tok"] for r in runs]), "cor": np.stack([r["cor"] for r in runs]),
            "trunc": np.stack([r["trunc"] for r in runs]), "uids": uids}


# ---- main ---------------------------------------------------------------------------------


def pilot(cfg: dict) -> None:
    root, pc = Path(cfg["results_dir"]), cfg["pilot"]
    run = load_run(root, "armA_random", 0)
    if run is None:
        raise SystemExit("pilot run armA_random/seed0 not fully evaluated yet")
    base = {}
    for es in EVAL_SETS:
        p = root / "_base" / "seed0" / "eval" / es / "summary.json"
        base[es] = json.loads(p.read_text()) if p.exists() else None
    res = {}
    for es in EVAL_SETS:
        s = run["summary"][es]
        res[es] = {"accuracy": s["accuracy"], "mean_tokens": s["mean_tokens"], "truncation_rate": s["truncation_rate"],
                   "base_accuracy": base[es]["accuracy"] if base[es] else None}
    print(json.dumps(pilot_gate(res, pc), indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.pilot:
        pilot(cfg)
        return
    acfg, root, data_dir = cfg["analysis"], Path(cfg["results_dir"]), Path(cfg["data_dir"])
    rng = np.random.default_rng(acfg["boot_seed"])

    missing, runs = [], {}
    wanted = [(n, s) for n in cfg["arm_a_sets"] for s in cfg["arm_a_seeds"]]
    wanted += [(n, cfg["arm_b_seed"]) for n in cfg["arm_b_sets"]]
    wanted += [(n, s) for n in cfg["arm_b_replicates"]["sets"] for s in cfg["arm_b_replicates"]["seeds"]]
    for n, s in wanted:
        r = load_run(root, n, s)
        if r is None:
            missing.append(f"{n}/seed{s}")
        else:
            runs[(n, s)] = r
    report: dict = {"experiment": cfg["experiment"], "missing_runs": missing, "n_runs": len(runs)}
    if not runs:
        print(json.dumps(report))
        return
    uids = next(iter(runs.values()))["uids"]
    n_q = len(uids)

    def rows(name):
        return [(s, r) for (n, s), r in sorted(runs.items()) if n == name]

    report["runs"] = {
        f"{n}/seed{s}": {
            "mean_tokens": float(r["tok"].mean()), "accuracy": float(r["cor"].mean()),
            "truncation_rate": float(r["trunc"].mean()),
            **{f"{es}_{k}": r["summary"][es][k] for es in EVAL_SETS for k in ("mean_tokens", "accuracy", "truncation_rate")},
        }
        for (n, s), r in sorted(runs.items())
    }

    # ---- Arm A contrasts ----------------------------------------------------------------------
    arms = {}
    for n in cfg["arm_a_sets"]:
        rs = rows(n)
        if len(rs) == len(cfg["arm_a_seeds"]):
            a = stack([r for _, r in rs])
            a["seeds"] = [s for s, _ in rs]
            arms[n] = a
    report["armA"] = {}
    for a, b in (("armA_high", "armA_low"), ("armA_high", "armA_random"), ("armA_low", "armA_random")):
        if a in arms and b in arms:
            report["armA"][f"{a}_vs_{b}"] = contrast(arms[a], arms[b], acfg["n_boot"], rng)

    # ---- Arm B variance decomposition ---------------------------------------------------------
    b_names = cfg["arm_b_sets"]
    have_b = all((n, cfg["arm_b_seed"]) in runs for n in b_names)
    group_toks = [np.stack([r["tok"] for _, r in rows(n)]) for n in list(arms) + cfg["arm_b_replicates"]["sets"]
                  if len(rows(n)) >= 3]
    groups = [g.mean(1) for g in group_toks]
    verdict = {"decision": "INCOMPLETE", "reason": f"missing runs: {len(missing)}"}
    if have_b and groups:
        tokB = np.stack([runs[(n, cfg["arm_b_seed"])]["tok"] for n in b_names])  # (16, Q)
        Y0 = tokB.mean(1)[None, :]
        noise, df = pooled_noise_var(groups)
        sources = sorted({r["source"] for n in b_names for r in read_jsonl(data_dir / f"train_{n}.jsonl")})
        feats = np.array([subset_features(data_dir, n, sources) for n in b_names])
        X = standardise(feats)
        point = variance_summary(Y0, noise, X, acfg["ridge_alphas"])
        from scipy.stats import f as fdist

        F = float(point["between_var"][0] / noise)
        p_val = float(fdist.sf(F, len(b_names) - 1, df))
        boot = {k: [] for k in ("explainable_frac", "r2_cv", "heuristic_share", "cost_range_rel")}
        draws = acfg["n_boot"]
        Yb, noise_b = np.empty((draws, len(b_names))), np.empty(draws)
        for d in range(draws):
            ix = rng.integers(0, n_q, n_q)
            Yb[d] = tokB[:, ix].mean(1)
            noise_b[d] = pooled_noise_var([g[:, ix].mean(1) for g in group_toks])[0]
        sb = variance_summary(Yb, noise_b, X, acfg["ridge_alphas"])
        for k in boot:
            vals = sb[k][~np.isnan(sb[k])]
            boot[k] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else None
        share = float(point["heuristic_share"][0]) if not np.isnan(point["heuristic_share"][0]) else 0.0
        decision, reason = decide(p_val, float(point["cost_range_rel"][0]), share, acfg)
        report["armB"] = {
            "n_subsets": len(b_names), "noise_var": noise, "noise_df": df, "between_var": float(point["between_var"][0]),
            "F": F, "F_p_value": p_val, "explainable_var": float(point["explainable_var"][0]),
            "explainable_frac": float(point["explainable_frac"][0]), "heuristic_r2_cv": float(point["r2_cv"][0]),
            "heuristic_share": share, "cost_range_rel": float(point["cost_range_rel"][0]),
            "subset_mean_tokens": {n: float(v) for n, v in zip(b_names, tokB.mean(1))},
            "bootstrap_ci95_prompts": boot, "n_features": int(X.shape[1]),
        }
        verdict = {"decision": decision, "reason": reason}
    report["verdict"] = verdict
    (root / "analysis.json").write_text(json.dumps(report, indent=2))
    md = render_md(report)
    (root / "analysis.md").write_text(md)
    print(md)


def render_md(rep: dict) -> str:
    L = [f"# {rep['experiment']} analysis", "", f"**Decision:** {rep['verdict']['decision']} ({rep['verdict']['reason']})", ""]
    if rep["missing_runs"]:
        L += [f"Missing runs: {', '.join(rep['missing_runs'])}", ""]
    if "armB" in rep:
        b = rep["armB"]
        L += ["## Arm B variance decomposition", "",
              f"- between-subset variance {b['between_var']:.1f}, seed-noise variance {b['noise_var']:.1f} (df {b['noise_df']})",
              f"- F = {b['F']:.2f}, one-sided p = {b['F_p_value']:.4f}",
              f"- explainable fraction {b['explainable_frac']:.3f}, cost range {100 * b['cost_range_rel']:.1f}% of the mean",
              f"- heuristic LOO R2 {b['heuristic_r2_cv']:.3f} ({b['n_features']} features), heuristic share {b['heuristic_share']:.3f}",
              f"- prompt-bootstrap 95% CIs: {json.dumps(b['bootstrap_ci95_prompts'])}", ""]
    if rep.get("armA"):
        L += ["## Arm A contrasts (tokens, hierarchical bootstrap)", "", "| contrast | delta | rel | 95% CI | seed-consistent | delta acc |", "|---|---|---|---|---|---|"]
        for k, c in rep["armA"].items():
            L.append(f"| {k} | {c['delta_tokens']:+.1f} | {100 * c['rel_delta_tokens']:+.1f}% | "
                     f"[{c['ci95_hierarchical'][0]:+.1f}, {c['ci95_hierarchical'][1]:+.1f}] | "
                     f"{c['sign_consistent_across_seeds']} | {c['delta_accuracy']:+.3f} |")
        L.append("")
    L += ["## Runs", "", "| run | mean tokens | accuracy | truncation | gsm8k tok | math500 tok |", "|---|---|---|---|---|---|"]
    for k, r in rep["runs"].items():
        L.append(f"| {k} | {r['mean_tokens']:.0f} | {r['accuracy']:.3f} | {r['truncation_rate']:.3f} | "
                 f"{r['gsm8k_test_mean_tokens']:.0f} | {r['math500_mean_tokens']:.0f} |")
    return "\n".join(L)


if __name__ == "__main__":
    main()
