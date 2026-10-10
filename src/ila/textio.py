"""Prompt format, answer extraction, and JSONL helpers shared by training and evaluation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from .arith import parse_number

PROMPT_TEMPLATE = "Question: {question}\nAnswer:\n"


def build_prompt(question: str, template: str | None = None) -> str:
    return (template or PROMPT_TEMPLATE).format(question=question.strip())


def build_chat_user_content(question: str, template: str | None = None) -> str:
    """Semantic prompt content for a canonical user chat turn.

    The plain-text ``Answer:`` cue is removed because the chat template opens the
    assistant turn explicitly.
    """
    return re.sub(r"\nAnswer:\s*$", "", build_prompt(question, template)).strip()


def extract_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (brace-matched), or None."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    depth, j = 1, i
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i : j - 1].strip() if depth == 0 else None


_ANSWER_RE = re.compile(r"The answer is\s*(.+?)\s*(?:\.\s*$|\n|$)", re.MULTILINE)
_NUM_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def extract_answer(text: str, fmt: str = "answer_is") -> str | None:
    """fmt='boxed': last \\boxed{} only. Otherwise last 'The answer is X', else the last number."""
    if fmt in ("boxed", "boxed_choice"):
        return extract_boxed(text)
    matches = _ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    nums = _NUM_RE.findall(text)
    return nums[-1] if nums else None


def _strip_latex(s: str) -> str:
    s = s.strip().strip("$").rstrip(".")
    m = re.fullmatch(r"\\boxed\{(.*)\}", s)
    if m:
        s = m.group(1)
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "")
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("\\%", "").replace("\\$", "")
    s = s.replace("\\,", "").replace("\\ ", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    return re.sub(r"\s+", "", s)


def answers_match(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    pn, gn = parse_number(pred), parse_number(gold)
    if pn is not None and gn is not None:
        return abs(pn - gn) <= 1e-6 * max(1.0, abs(gn))
    return _strip_latex(pred) == _strip_latex(gold)


def answers_match_choice(pred: str | None, gold: str) -> bool:
    """Normalize only an explicit option letter; never guess from a longer response."""
    if pred is None or gold.upper() not in "ABCDE" or len(gold) != 1:
        return False
    normalized = _strip_latex(pred).upper()
    normalized = re.sub(r"\\(?:MATHRM|MATHBF)\{([A-E])\}", r"\1", normalized)
    match = re.fullmatch(r"[\(\[]?([A-E])[\)\]]?", normalized)
    return bool(match and match[1] == gold.upper())


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
