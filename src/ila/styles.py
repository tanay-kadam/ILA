"""Programmatic reasoning-style variants of a GSM8K solution.

Every variant of a problem is arithmetically correct and ends in the same final line.
Each non-concise style exposes a ladder of *levels* (increasing number of insertions) so
that variants of different styles can be length-matched per problem.

Styles:
  concise    - original GSM8K steps.
  detailed   - linear, forward-only elaboration: step labels, restated givens,
               place-value breakdowns. Never revisits an earlier computation.
  verify     - after steps, inverse-operation checks, then repeated re-checks.
  backtrack  - before steps, an erroneous attempt that is explicitly abandoned.
  resolve    - after the full solution, a redundant recomputation of all steps.

Phrase pools: index 0 of every pool is the original fixed wording. With para=False only
index 0 is used (identical to the first experiment). With para=True each slot picks a
variant deterministically from (problem uid, style, slot, position), so a given insertion
keeps its wording across levels and no single phrase identifies a style.
"""

from __future__ import annotations

import random
import zlib
from typing import Callable

from .arith import decomposition, display_expr, fmt, inverse_check, wrong_attempts
from .gsm8k import Problem, question_facts

MATCHED_STYLES = ("detailed", "verify", "backtrack", "resolve")
ALL_STYLES = ("concise",) + MATCHED_STYLES

POOLS: dict[str, list[str]] = {
    "verify_first": [
        "Let me double-check: {x}. Yes, that is correct.",
        "Verifying: {x}. That checks out.",
        "Quick sanity check: {x}. Good, that holds.",
        "To be safe, I'll confirm: {x}. Correct.",
        "Checking this step: {x}. It is right.",
    ],
    "verify_second": [
        "Let me check that once more: {x}. Yes, that's right.",
        "Confirming again: {x}. Still right.",
        "One more verification: {x}. That's consistent.",
        "Rechecking: {x}. No error.",
    ],
    "backtrack_first": [
        "First I'll try {x}. Wait, that's not right. Let me redo this step.",
        "Let me guess {x}. No, that's incorrect. Starting this step over.",
        "Trying {x}. Hmm, that can't be right. Let me redo it.",
        "Initially I think {x}. Actually, that's wrong. Let me recompute.",
        "I'll start with {x}. On reflection, that's a mistake. Let me redo this.",
    ],
    "backtrack_second": [
        "Maybe {x}? Hmm, that doesn't work either.",
        "Perhaps {x}? No, that fails as well.",
        "What about {x}? That isn't it either.",
    ],
    "resolve_intro": [
        "Let me solve this again another way to make sure.",
        "To be certain, I'll work through it a second time.",
        "I want to confirm, so here is the solution again.",
        "Let me redo the whole calculation from the start.",
    ],
    "resolve_line": ["We get {x}.", "Next, {x}.", "This gives {x}.", "So {x}."],
    "resolve_agree": ["Both solutions give {x}.", "The second pass also gives {x}.", "This matches: {x}."],
    "resolve_third": [
        "Let me go through it one more time.",
        "Let me run through it a third time.",
        "I'll go over it once more.",
    ],
    "resolve_again": ["Again, {x}.", "Once more, {x}.", "Same result: {x}."],
    "resolve_all": ["All solutions agree.", "All passes agree.", "Everything is consistent."],
    "detailed_given": ["Given information:", "Known facts:", "What we know:", "Key facts:"],
    "detailed_now": [
        "Now let's solve it step by step.",
        "Let's work through it.",
        "Now the solution:",
        "We proceed as follows.",
    ],
    "detailed_step": ["Step {x}:", "({x})", "{x}.", "Part {x}:"],
    "detailed_break": ["Breaking this down:", "Splitting it up:", "In pieces:"],
    "detailed_restate": [
        "Let's first restate the problem:",
        "To begin, here is the problem again:",
        "The problem, restated:",
    ],
}

# Surface markers used to measure style transfer in model outputs: every phrase variant
# that is distinctive for its style (original wording first).
MARKERS = {
    "detailed": [
        "Given information", "Known facts", "What we know", "Key facts", "Breaking this down",
        "Splitting it up", "In pieces", "Step 1:", "Part 1:",
    ],
    "verify": [
        "double-check", "check that once more", "Verifying:", "sanity check", "I'll confirm",
        "Checking this step", "Confirming again", "One more verification", "Rechecking:",
    ],
    "backtrack": [
        "that's not right", "doesn't work either", "that's incorrect", "can't be right",
        "that's wrong", "that's a mistake", "fails as well", "isn't it either",
    ],
    "resolve": [
        "solve this again", "one more time", "second time", "here is the solution again",
        "redo the whole calculation", "third time", "go over it once more", "All passes agree",
        "Everything is consistent", "The second pass also gives",
    ],
}


def _final(p: Problem) -> str:
    return f"The answer is {fmt(p.answer)}."


def _rng(p: Problem, style: str) -> random.Random:
    return random.Random(zlib.crc32(f"{p.uid}|{style}".encode()))


def pick(p: Problem, para: bool, pool: str, idx: int = 0) -> str:
    options = POOLS[pool]
    if not para:
        return options[0]
    return options[zlib.crc32(f"{p.uid}|{pool}|{idx}".encode()) % len(options)]


def concise(p: Problem) -> str:
    return "\n".join([s.text for s in p.steps] + [_final(p)])


def _calc_line(expr: str, res: float) -> str:
    return f"{display_expr(expr)} = {fmt(res)}"


def verify_levels(p: Problem, para: bool = False) -> list[str]:
    eligible = p.calc_steps
    order = eligible[:]
    _rng(p, "verify").shuffle(order)
    n = len(order)
    levels = []
    for k in range(1, 2 * n + 1):
        first, second = set(order[: min(k, n)]), set(order[: max(0, k - n)])
        lines = []
        for i, s in enumerate(p.steps):
            lines.append(s.text)
            if i in first:
                c = s.calcs[0]
                chk = inverse_check(c.expr, c.result) or _calc_line(c.expr, c.result)
                lines.append(pick(p, para, "verify_first", i).format(x=chk))
            if i in second:
                c = s.calcs[0]
                lines.append(pick(p, para, "verify_second", i).format(x=_calc_line(c.expr, c.result)))
        levels.append("\n".join(lines + [_final(p)]))
    return levels


def backtrack_levels(p: Problem, para: bool = False) -> list[str]:
    wrongs = {i: wrong_attempts(p.steps[i].calcs[0].expr, p.steps[i].calcs[0].result) for i in p.calc_steps}
    first_ok = [i for i in p.calc_steps if len(wrongs[i]) >= 1]
    second_ok = [i for i in p.calc_steps if len(wrongs[i]) >= 2]
    rng = _rng(p, "backtrack")
    rng.shuffle(first_ok)
    rng.shuffle(second_ok)
    n1, n2 = len(first_ok), len(second_ok)
    levels = []
    for k in range(1, n1 + n2 + 1):
        first = set(first_ok[: min(k, n1)])
        second = set(i for i in second_ok[: max(0, k - n1)] if i in first)
        lines = []
        for i, s in enumerate(p.steps):
            if i in first:
                we, wv = wrongs[i][0]
                lines.append(pick(p, para, "backtrack_first", i).format(x=f"{we} = {fmt(wv)}"))
            if i in second:
                we, wv = wrongs[i][1]
                lines.append(pick(p, para, "backtrack_second", i).format(x=f"{we} = {fmt(wv)}"))
            lines.append(s.text)
        levels.append("\n".join(lines + [_final(p)]))
    return levels


def resolve_levels(p: Problem, para: bool = False) -> list[str]:
    calcs = [c for s in p.steps for c in s.calcs]
    n = len(calcs)
    base = [s.text for s in p.steps]
    levels = []
    for k in range(1, 2 * n + 1):
        lines = base + [pick(p, para, "resolve_intro")]
        lines += [
            pick(p, para, "resolve_line", j).format(x=_calc_line(c.expr, c.result)) for j, c in enumerate(calcs[: min(k, n)])
        ]
        if k >= n:
            lines.append(pick(p, para, "resolve_agree").format(x=fmt(p.answer)))
        if k > n:
            lines.append(pick(p, para, "resolve_third"))
            lines += [
                pick(p, para, "resolve_again", j).format(x=_calc_line(c.expr, c.result))
                for j, c in enumerate(calcs[: k - n])
            ]
            if k == 2 * n:
                lines.append(pick(p, para, "resolve_all"))
        levels.append("\n".join(lines + [_final(p)]))
    return levels


def detailed_levels(p: Problem, para: bool = False) -> list[str]:
    facts = question_facts(p.question)
    decomps: dict[int, str] = {}
    for i, s in enumerate(p.steps):
        for c in s.calcs:
            d = decomposition(c.expr, c.result)
            if d:
                decomps[i] = d.replace("Breaking this down:", pick(p, para, "detailed_break", i), 1)
                break
    dec_order = sorted(decomps)
    n_f, n_d = len(facts), len(dec_order)
    levels = []
    for k in range(0, n_f + n_d + 2):
        n_facts = min(k, n_f)
        use_dec = set(dec_order[: max(0, min(k - n_f, n_d))])
        restate = k == n_f + n_d + 1
        lines = []
        if restate:
            lines.append(f"{pick(p, para, 'detailed_restate')} {p.question}")
        if n_facts:
            lines.append(pick(p, para, "detailed_given"))
            lines += [f"- {f}" for f in facts[:n_facts]]
            lines.append(pick(p, para, "detailed_now"))
        step_tpl = pick(p, para, "detailed_step")
        for j, s in enumerate(p.steps):
            lines.append(f"{step_tpl.format(x=j + 1)} {s.text}")
            if j in use_dec:
                lines.append(decomps[j])
        levels.append("\n".join(lines + [_final(p)]))
    return levels


LEVEL_FNS: dict[str, Callable[..., list[str]]] = {
    "detailed": detailed_levels,
    "verify": verify_levels,
    "backtrack": backtrack_levels,
    "resolve": resolve_levels,
}


def length_matched_variants(
    p: Problem,
    length_fn: Callable[[str], int],
    min_ratio: float,
    tol: float,
    para: bool = False,
) -> dict | None:
    """Pick one level per matched style so all are within `tol` relative length of a common
    per-problem target T >= min_ratio * len(concise). Returns None if impossible."""
    conc = concise(p)
    conc_len = length_fn(conc)
    ladders: dict[str, list[tuple[int, str]]] = {}
    for style, fn in LEVEL_FNS.items():
        lv = fn(p, para)
        if not lv:
            return None
        ladders[style] = [(length_fn(t), t) for t in lv]
    candidates = sorted({L for lad in ladders.values() for L, _ in lad if L >= min_ratio * conc_len})
    best = None
    for T in candidates:
        picks, dev = {}, 0.0
        for style, lad in ladders.items():
            idx = min(range(len(lad)), key=lambda j: abs(lad[j][0] - T))
            picks[style] = idx
            dev = max(dev, abs(lad[idx][0] - T) / T)
        if best is None or dev < best[0] - 1e-12:
            best = (dev, T, picks)
    if best is None or best[0] > tol:
        return None
    dev, T, picks = best
    out = {"concise": {"text": conc, "tokens": conc_len, "level": 0}}
    for style, idx in picks.items():
        L, t = ladders[style][idx]
        out[style] = {"text": t, "tokens": L, "level": idx}
    out["_meta"] = {"target": T, "max_rel_dev": dev}
    return out
