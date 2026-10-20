import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ila.arith import safe_eval  # noqa: E402
from ila.gsm8k import parse_solution  # noqa: E402
from ila.styles import ALL_STYLES, LEVEL_FNS, MARKERS, POOLS, concise, length_matched_variants  # noqa: E402
from ila.textio import extract_answer  # noqa: E402

CASES = [
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
        "How many clips did Natalia sell altogether in April and May?",
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
        "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
    ),
    (
        "A store sells 37 boxes with 23 pens each. It then sells 115 more pens. How many pens were sold?",
        "Boxes hold 37*23 = <<37*23=851>>851 pens.\nIn total 851+115 = <<851+115=966>>966 pens.\n#### 966",
    ),
    (
        "Tom buys 14 apples at 3 dollars each and 6 pears at 5 dollars each. How much does he pay?",
        "Apples cost 14*3 = <<14*3=42>>42 dollars.\nPears cost 6*5 = <<6*5=30>>30 dollars.\n"
        "He pays 42+30 = <<42+30=72>>72 dollars.\n#### 72",
    ),
]
PROBLEMS = [parse_solution(f"t{i}", q, a) for i, (q, a) in enumerate(CASES)]
EQ = re.compile(r"(-?\d+(?:\.\d+)?(?: [-+*/] -?\d+(?:\.\d+)?)+) = (-?\d+(?:\.\d+)?)")
WRONG_PHRASES = [t.split("{x}")[0] for t in POOLS["backtrack_first"] + POOLS["backtrack_second"]]


def ws_len(t: str) -> int:
    return len(t.split())


@pytest.fixture(params=PROBLEMS, ids=lambda p: p.uid)
def problem(request):
    return request.param


def test_default_matches_original_wording(problem):
    for fn in LEVEL_FNS.values():
        assert fn(problem) == fn(problem, False)
    assert "Let me double-check" in LEVEL_FNS["verify"](problem)[0]


def test_para_same_final_answer(problem):
    final = concise(problem).splitlines()[-1]
    for fn in LEVEL_FNS.values():
        for t in fn(problem, True):
            assert t.splitlines()[-1] == final
            assert extract_answer(t) == final.removeprefix("The answer is ").rstrip(".")


def test_para_lengths_monotone(problem):
    for style, fn in LEVEL_FNS.items():
        lens = [ws_len(t) for t in fn(problem, True)]
        assert lens == sorted(lens), style


def test_para_non_backtrack_equations_true(problem):
    for style in ("detailed", "verify", "resolve"):
        for t in LEVEL_FNS[style](problem, True):
            for lhs, rhs in EQ.findall(t):
                assert abs(safe_eval(lhs) - float(rhs)) < 1e-6, (style, lhs, rhs)


def test_para_backtrack_attempts_are_wrong_equations(problem):
    seen = 0
    for t in LEVEL_FNS["backtrack"](problem, True):
        for line in t.splitlines():
            if any(line.startswith(ph) for ph in WRONG_PHRASES):
                lhs, rhs = EQ.findall(line)[0]
                assert abs(safe_eval(lhs) - float(rhs)) < 1e-6
                seen += 1
    assert seen > 0


def test_no_marker_leakage_across_styles(problem):
    for style, fn in LEVEL_FNS.items():
        for para in (False, True):
            for t in fn(problem, para):
                for other, marks in MARKERS.items():
                    if other == style:
                        continue
                    for m in marks:
                        assert m not in t, (style, para, other, m)
    for m in sum(MARKERS.values(), []):
        assert m not in concise(problem), m


def test_each_style_contains_own_marker(problem):
    for style, fn in LEVEL_FNS.items():
        for para in (False, True):
            for t in fn(problem, para):
                assert any(m in t for m in MARKERS[style]) or style == "detailed", (style, para)
    assert all(any(m in t for m in MARKERS["detailed"]) for t in LEVEL_FNS["detailed"](problem, True)[1:])


def test_para_uses_multiple_variants():
    texts = set()
    for i in range(40):
        p = parse_solution(f"u{i}", *CASES[2])
        texts.add(LEVEL_FNS["verify"](p, True)[0].splitlines()[1])
    assert len(texts) >= 3


def test_para_length_matching(problem):
    v = length_matched_variants(problem, ws_len, min_ratio=1.3, tol=0.30, para=True)
    assert v is not None
    T = v["_meta"]["target"]
    for s in ALL_STYLES[1:]:
        assert abs(v[s]["tokens"] - T) / T <= 0.30
