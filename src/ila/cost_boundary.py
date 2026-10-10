"""Descriptive capped-cost decomposition, not causal mediation."""
from __future__ import annotations
import numpy as np


def measures(tokens, truncated, correct=None, cap=6144):
    t = np.asarray(tokens, dtype=float)
    q = np.asarray(truncated, dtype=bool)
    if not len(t) or len(t) != len(q) or np.any(t < 0) or np.any(t > cap):
        raise ValueError("invalid token/truncation arrays")
    if np.any(q & (t != cap)):
        raise ValueError("a cap-hit row must contain exactly cap tokens")
    finished = t[~q]
    result = {"n": len(t), "mean_tokens": float(t.mean()), "truncation_rate": float(q.mean()),
              "finished_mean_tokens": float(finished.mean()) if len(finished) else None,
              "capped_row_contribution": float(t[q].sum() / len(t)),
              "finished_row_contribution": float(finished.sum() / len(t))}
    if correct is not None:
        c = np.asarray(correct, dtype=bool)
        if len(c) != len(t):
            raise ValueError("correctness array mismatch")
        result.update(accuracy=float(c.mean()), tokens_per_correct=float(t.sum() / c.sum()) if c.sum() else None)
    return result


def decompose(a, b, cap=6144):
    delta = a["mean_tokens"] - b["mean_tokens"]
    direct = {"delta_tokens": delta,
              "capped_row_difference": a["capped_row_contribution"] - b["capped_row_contribution"],
              "finished_row_difference": a["finished_row_contribution"] - b["finished_row_contribution"]}
    ua, ub = a["finished_mean_tokens"], b["finished_mean_tokens"]
    if ua is None or ub is None:
        return {**direct, "symmetric_identified": False}
    qa, qb = a["truncation_rate"], b["truncation_rate"]
    frequency = (qa - qb) * (cap - (ua + ub) / 2)
    within = (ua - ub) * (1 - (qa + qb) / 2)
    if not np.isclose(frequency + within, delta, atol=1e-8, rtol=1e-10):
        raise ValueError("decomposition identity failed")
    return {**direct, "symmetric_identified": True, "frequency_component": frequency,
            "finished_length_component": within,
            "frequency_dominates_descriptively": bool(np.sign(frequency) == np.sign(delta) and abs(frequency) > abs(within))}


def bootstrap(a, b, draws=2000, seed=20261007, paired=False, cap=6144):
    """Same prompt index for both arms; independent or paired seed blocks."""
    rng = np.random.default_rng(seed)
    n_seeds, n_q = a["tokens"].shape
    if a["tokens"].shape != b["tokens"].shape:
        raise ValueError("aligned arm arrays required")
    values = {k: [] for k in ("delta_tokens", "frequency_component", "finished_length_component", "accuracy_delta", "truncation_delta")}
    for _ in range(draws):
        ix = rng.integers(0, n_q, n_q)
        sa = rng.integers(0, n_seeds, n_seeds)
        sb = sa if paired else rng.integers(0, n_seeds, n_seeds)
        aa = measures(a["tokens"][sa][:, ix].ravel(), a["truncated"][sa][:, ix].ravel(), a["correct"][sa][:, ix].ravel(), cap)
        bb = measures(b["tokens"][sb][:, ix].ravel(), b["truncated"][sb][:, ix].ravel(), b["correct"][sb][:, ix].ravel(), cap)
        d = decompose(aa, bb, cap)
        d.update(accuracy_delta=aa["accuracy"]-bb["accuracy"], truncation_delta=aa["truncation_rate"]-bb["truncation_rate"])
        for k in values:
            if k in d:
                values[k].append(d[k])
    return {k: {"ci95": [float(x) for x in np.quantile(v, [.025, .975])], "identified_draws": len(v)}
            for k, v in values.items() if v}
