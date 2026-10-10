"""Cheap per-trace heuristics for natural reasoning traces.

These are the baselines a data engineer could compute with a regex. Stage 1 measures how
much of the training-data -> inference-cost effect they explain.
"""

from __future__ import annotations

import re

REFLECTION_PATTERNS = [
    r"\bwait\b",
    r"let me (?:re)?check",
    r"let me (?:re)?verify",
    r"double[- ]check",
    r"\balternatively\b",
    r"\bhmm\b",
]
RESTART_PATTERNS = [
    r"start over",
    r"let me try again",
    r"let's try again",
    r"try a different",
    r"\bre-?think\b",
    r"\bre-?consider\b",
    r"that doesn't seem right",
    r"that can't be",
]
_REFLECTION_RE = re.compile("|".join(REFLECTION_PATTERNS), re.IGNORECASE)
_RESTART_RE = re.compile("|".join(RESTART_PATTERNS), re.IGNORECASE)

# Marker strings counted in model outputs at evaluation time (see configs/stage1.yaml).
EVAL_MARKERS = {
    "reflection": [
        "Wait", "wait", "Let me check", "let me check", "Let me verify", "let me verify",
        "double-check", "double check", "Alternatively", "alternatively", "Hmm", "hmm",
    ],
    "restart": [
        "start over", "let me try again", "let's try again", "try a different", "rethink",
        "reconsider", "That doesn't seem right", "that can't be",
    ],
}

TRACE_FEATURES = ("n_tokens", "reflection_density", "restart_count", "boxed_count")


def count_reflections(text: str) -> int:
    return len(_REFLECTION_RE.findall(text))


def count_restarts(text: str) -> int:
    return len(_RESTART_RE.findall(text))


def count_boxed(text: str) -> int:
    return text.count("\\boxed{")


def trace_features(text: str, n_tokens: int) -> dict:
    return {
        "n_tokens": n_tokens,
        "reflection_density": 1000.0 * count_reflections(text) / max(n_tokens, 1),
        "restart_count": count_restarts(text),
        "boxed_count": count_boxed(text),
    }
