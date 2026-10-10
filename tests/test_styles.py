import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ila.arith import safe_eval, wrong_attempts, inverse_check, decomposition  # noqa: E402
from ila.gsm8k import ParseError, parse_solution  # noqa: E402
from ila.styles import (  # noqa: E402
    ALL_STYLES,
    LEVEL_FNS,
    concise,
    length_matched_variants,
)
from ila.textio import answers_match, extract_answer  # noqa: E402

Q = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?"
)
A = (
    "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
)
Q2 = "A store sells 37 boxes with 23 pens each. It then sells 115 more pens. How many pens were sold?"
A2 = "Boxes hold 37*23 = <<37*23=851>>851 pens.\nIn total 851+115 = <<851+115=966>>966 pens.\n#### 966"


def ws_len(t: str) -> int:
    return len(t.split())


@pytest.fixture(params=[(Q, A), (Q2, A2)])
def problem(request):
    return parse_solution("t", *request.param)


def test_parse(problem):
    assert problem.calc_steps
    assert "<<" not in concise(problem)


def test_parse_rejects_bad_annotation():
    with pytest.raises(ParseError):
        parse_solution("bad", Q, "x 2+2 = <<2+2=5>>5\n#### 5")


def test_every_level_ends_with_same_answer(problem):
    final = concise(problem).splitlines()[-1]
    for style, fn in LEVEL_FNS.items():
        levels = fn(problem)
        assert levels, style
        for t in levels:
            assert t.splitlines()[-1] == final
            assert extract_answer(t) == final.removeprefix("The answer is ").rstrip(".")


def test_levels_monotone_in_length(problem):
    for style, fn in LEVEL_FNS.items():
        lens = [ws_len(t) for t in fn(problem)]
        assert lens == sorted(lens), style


def _stated_equations(text: str):
    return re.findall(r"(-?\d+(?:\.\d+)?(?: [-+*/] -?\d+(?:\.\d+)?)+) = (-?\d+(?:\.\d+)?)", text)


def test_non_backtrack_equations_are_true(problem):
    for style in ("detailed", "verify", "resolve"):
        for t in LEVEL_FNS[style](problem):
            for lhs, rhs in _stated_equations(t):
                assert abs(safe_eval(lhs) - float(rhs)) < 1e-6, (style, lhs, rhs)


def test_backtrack_wrong_attempts_are_wrong_and_flagged(problem):
    for t in LEVEL_FNS["backtrack"](problem):
        for line in t.splitlines():
            if line.startswith("First I'll try") or line.startswith("Maybe"):
                lhs, rhs = _stated_equations(line)[0]
                assert abs(safe_eval(lhs) - float(rhs)) < 1e-6
                assert "not right" in line or "either" in line


def test_detailed_never_revisits(problem):
    for t in LEVEL_FNS["detailed"](problem):
        for m in ("check", "Wait", "again", "redo"):
            assert m not in t


def test_arith_helpers():
    assert inverse_check("48/2", 24) == "24 * 2 = 48"
    assert all(v != 72 for _, v in wrong_attempts("48+24", 72))
    assert decomposition("37*23", 851) == "Breaking this down: 37 * 20 = 740 and 37 * 3 = 111, so 740 + 111 = 851."


def test_length_matching(problem):
    v = length_matched_variants(problem, ws_len, min_ratio=1.3, tol=0.25)
    assert v is not None
    T = v["_meta"]["target"]
    for s in ALL_STYLES[1:]:
        assert abs(v[s]["tokens"] - T) / T <= 0.25
    assert T >= 1.3 * v["concise"]["tokens"]


def test_answer_matching():
    assert answers_match("1,000", "1000")
    assert answers_match("\\frac{1}{2}", "\\dfrac{1}{2}")
    assert not answers_match(None, "3")
    assert extract_answer("blah\nThe answer is 42.") == "42"
