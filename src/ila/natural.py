"""Pool filtering, decontamination, and subset construction for the Stage 1 natural-data gate."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

_WORD = re.compile(r"\w+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def ngram_set(texts: Iterable[str], n: int = 13) -> tuple[set, set]:
    """Return (all n-grams, exact word sequences of texts shorter than n words)."""
    grams, short = set(), set()
    for t in texts:
        w = words(t)
        if len(w) < n:
            short.add(tuple(w))
            continue
        for i in range(len(w) - n + 1):
            grams.add(tuple(w[i : i + n]))
    return grams, short


def is_contaminated(problem: str, grams: set, short: set, n: int = 13) -> bool:
    w = words(problem)
    if len(w) < n:
        return tuple(w) in short or tuple(w) in grams
    return any(tuple(w[i : i + n]) in grams for i in range(len(w) - n + 1))


def pick_trace(row: dict) -> str | None:
    """First generation that is verified correct, reasoning-complete and stop-terminated."""
    gens = row.get("generations") or []
    correct = row.get("correctness_math_verify") or []
    complete = row.get("is_reasoning_complete") or []
    finish = row.get("finish_reasons")
    for i, g in enumerate(gens):
        if i >= len(correct) or not correct[i]:
            continue
        if i >= len(complete) or not complete[i]:
            continue
        if finish is not None and (i >= len(finish) or finish[i] != "stop"):
            continue
        g = (g or "").strip()
        if g:
            return g
    return None


def apportion(weights: Sequence[float], total: int, caps: Sequence[int]) -> np.ndarray:
    """Integer allocation proportional to weights, never exceeding caps (largest remainder).

    Excess from capped entries is redistributed over the rest. If sum(caps) < total the
    allocation is partial.
    """
    w = np.asarray(weights, dtype=float)
    caps = np.asarray(caps, dtype=int)
    alloc = np.zeros(len(w), dtype=int)
    remaining = int(total)
    active = (caps > 0) & (w > 0)
    while remaining > 0 and active.any():
        wa = np.where(active, w, 0.0)
        share = wa / wa.sum() * remaining
        base = np.minimum(np.floor(share).astype(int), caps - alloc)
        alloc += base
        remaining -= int(base.sum())
        if remaining > 0:
            frac = share - np.floor(share)
            frac[~active] = -1.0
            frac[alloc >= caps] = -1.0
            for i in np.argsort(-frac, kind="stable"):
                if remaining == 0:
                    break
                if frac[i] >= 0 and alloc[i] < caps[i]:
                    alloc[i] += 1
                    remaining -= 1
        active = (alloc < caps) & (w > 0)
    return alloc


def length_deciles(n_tokens: Sequence[float]) -> np.ndarray:
    edges = np.quantile(np.asarray(n_tokens, dtype=float), np.linspace(0, 1, 11)[1:-1])
    return np.searchsorted(edges, np.asarray(n_tokens, dtype=float), side="right")


def matched_arms(items: Sequence[dict], n_total: int, rng: np.random.Generator, min_bin: int = 6) -> dict:
    """Length/source/type-matched high vs low reflection density sets, plus a random control.

    Bins are length decile x source x problem type. Each bin of at least `min_bin` traces
    contributes the same number of traces to all three sets, so the sets have identical bin
    composition. `high`/`low` take the extremes of reflection density inside the bin and
    `random` samples the bin uniformly.
    """
    deciles = length_deciles([it["n_tokens"] for it in items])
    bins: dict[tuple, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        bins[(int(deciles[i]), it["source"], it["problem_type"])].append(i)
    keys = sorted(bins, key=str)
    quota = np.array([len(bins[k]) // 3 if len(bins[k]) >= min_bin else 0 for k in keys])
    alloc = quota.copy() if quota.sum() <= n_total else apportion(quota, n_total, quota)

    high, low, rand = [], [], []
    for k, a in zip(keys, alloc):
        if a == 0:
            continue
        members = bins[k]
        tiebreak = rng.random(len(members))
        order = sorted(range(len(members)), key=lambda j: (items[members[j]]["reflection_density"], tiebreak[j]))
        low += [members[j] for j in order[:a]]
        high += [members[j] for j in order[len(order) - a :]]
        rand += [members[j] for j in rng.choice(len(members), size=a, replace=False)]
    return {"high": high, "low": low, "random": rand, "n_bins_used": int((alloc > 0).sum()), "n_bins": len(keys)}


def cluster_problems(texts: Sequence[str], k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = TfidfVectorizer(max_features=20000, stop_words="english", sublinear_tf=True).fit_transform(texts)
    return KMeans(n_clusters=k, n_init=3, random_state=seed).fit_predict(X)


def dirichlet_subsets(
    labels: np.ndarray, n_subsets: int, n_per: int, alpha: float, rng: np.random.Generator
) -> tuple[list[list[int]], list[list[float]]]:
    """Subsets whose cluster mix ~ Dirichlet(alpha), capped at cluster size.

    Returns (indices per subset, realised cluster fractions per subset).
    """
    k = int(labels.max()) + 1
    members = [np.flatnonzero(labels == c) for c in range(k)]
    sizes = np.array([len(m) for m in members])
    subsets, mixes = [], []
    for _ in range(n_subsets):
        w = rng.dirichlet(np.full(k, alpha))
        counts = apportion(w, n_per, sizes)
        idx: list[int] = []
        for c in range(k):
            if counts[c]:
                idx += rng.choice(members[c], size=counts[c], replace=False).tolist()
        subsets.append(idx)
        mixes.append((counts / max(counts.sum(), 1)).tolist())
    return subsets, mixes
