"""Parsing GSM8K solutions into steps with verified calculator annotations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .arith import ArithError, parse_number, safe_eval

_ANNOT = re.compile(r"<<([^=<>]+)=([^<>]+)>>")


@dataclass
class Calc:
    expr: str
    result: float


@dataclass
class Step:
    text: str
    calcs: list[Calc] = field(default_factory=list)


@dataclass
class Problem:
    uid: str
    question: str
    steps: list[Step]
    answer: float

    @property
    def calc_steps(self) -> list[int]:
        return [i for i, s in enumerate(self.steps) if s.calcs]


class ParseError(ValueError):
    pass


def parse_solution(uid: str, question: str, solution: str) -> Problem:
    if "####" not in solution:
        raise ParseError("missing final answer")
    body, final = solution.rsplit("####", 1)
    answer = parse_number(final)
    if answer is None:
        raise ParseError(f"non-numeric answer {final!r}")
    steps: list[Step] = []
    for raw in body.strip().split("\n"):
        raw = raw.strip()
        if not raw:
            continue
        calcs = []
        for m in _ANNOT.finditer(raw):
            expr, res_s = m.group(1), m.group(2)
            res = parse_number(res_s)
            if res is None:
                raise ParseError(f"bad annotation result {res_s!r}")
            try:
                val = safe_eval(expr)
            except ArithError as e:
                raise ParseError(str(e)) from e
            if abs(val - res) > 1e-2 * max(1.0, abs(res)):
                raise ParseError(f"annotation mismatch {expr}={res_s} (eval {val})")
            calcs.append(Calc(expr.strip(), res))
        text = re.sub(r"\s+", " ", _ANNOT.sub("", raw)).strip()
        steps.append(Step(text, calcs))
    if not steps:
        raise ParseError("empty solution")
    return Problem(uid, question.strip(), steps, answer)


def question_facts(question: str) -> list[str]:
    """Sentences of the question that carry numeric information."""
    sents = re.split(r"(?<=[.!?])\s+", question.strip())
    return [s for s in sents if re.search(r"\d", s) and not s.rstrip().endswith("?")]
