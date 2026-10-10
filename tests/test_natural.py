import numpy as np
import pytest

from ila.features import count_boxed, count_reflections, count_restarts, trace_features
from ila.natural import (
    apportion,
    cluster_problems,
    dirichlet_subsets,
    is_contaminated,
    matched_arms,
    ngram_set,
    pick_trace,
)


def test_reflection_and_restart_counts():
    t = "Okay. Wait, that's wrong. Hmm. Let me check again. Alternatively, double-check. Wait."
    assert count_reflections(t) == 6
    assert count_reflections("The waiter waited.") == 0
    assert count_restarts("Let me try again. I should start over.") == 2
    assert count_boxed("so \\boxed{3} and again \\boxed{3}") == 2


def test_trace_features_density():
    f = trace_features("wait wait wait", 1500)
    assert f["reflection_density"] == pytest.approx(2.0)
    assert f["n_tokens"] == 1500


def test_pick_trace_requires_correct_complete_stop():
    row = {
        "generations": ["a", "b", "c"],
        "correctness_math_verify": [False, True, True],
        "is_reasoning_complete": [True, False, True],
        "finish_reasons": ["stop", "stop", "stop"],
    }
    assert pick_trace(row) == "c"
    row["finish_reasons"] = ["stop", "stop", "length"]
    assert pick_trace(row) is None
    row["finish_reasons"] = None
    assert pick_trace(row) == "c"


def test_decontamination_ngram_and_short():
    grams, short = ngram_set(["one two three four five six seven eight nine ten eleven twelve thirteen fourteen", "What is 2+2"], 13)
    long_hit = "Intro. two three four five six seven eight nine ten eleven twelve thirteen fourteen more"
    assert is_contaminated(long_hit, grams, short)
    assert not is_contaminated("totally different text " * 10, grams, short)
    assert is_contaminated("what is 2 2", grams, short)
    assert not is_contaminated("what is 3 3", grams, short)


def test_apportion_respects_caps_and_total():
    a = apportion([0.7, 0.2, 0.1], 100, [30, 100, 100])
    assert a.sum() == 100 and a[0] == 30
    assert a[1] > a[2]
    p = apportion([1, 1], 10, [3, 4])
    assert p.sum() == 7


def _items(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {
            "n_tokens": int(rng.integers(1000, 4000)),
            "source": f"s{rng.integers(0, 3)}",
            "problem_type": f"t{rng.integers(0, 2)}",
            "reflection_density": float(rng.gamma(2.0, 2.0)),
        }
        for _ in range(n)
    ]


def test_matched_arms_composition_and_contrast():
    items = _items()
    arms = matched_arms(items, 600, np.random.default_rng(1))
    hi, lo, rd = arms["high"], arms["low"], arms["random"]
    assert len(hi) == len(lo) == len(rd) == 600
    assert set(hi).isdisjoint(lo)
    dens = lambda ix: np.mean([items[i]["reflection_density"] for i in ix])
    assert dens(hi) > dens(rd) > dens(lo)
    ln = lambda ix: np.mean([items[i]["n_tokens"] for i in ix])
    assert abs(ln(hi) / ln(lo) - 1) < 0.03

    def comp(ix):
        from collections import Counter

        return Counter((items[i]["source"], items[i]["problem_type"]) for i in ix)

    assert comp(hi) == comp(lo) == comp(rd)


def test_cluster_and_dirichlet_subsets():
    rng = np.random.default_rng(0)
    topics = ["algebra equation solve polynomial root", "triangle angle circle geometry radius",
              "probability dice coin random expected", "integer prime divisor modulo remainder"]
    texts = [f"{topics[i % 4]} problem {rng.integers(0, 50)}" for i in range(400)]
    labels = cluster_problems(texts, 4, 0)
    assert len(set(labels.tolist())) == 4
    subsets, mixes = dirichlet_subsets(labels, 5, 60, 0.5, np.random.default_rng(0))
    assert len(subsets) == 5
    for s, m in zip(subsets, mixes):
        assert len(s) == len(set(s)) == 60
        assert sum(m) == pytest.approx(1.0)
    assert subsets[0] != subsets[1]


def test_extract_boxed_nested_and_match():
    from ila.textio import answers_match, build_prompt, extract_answer, extract_boxed

    assert extract_boxed("so \\boxed{1} then \\boxed{\\frac{3}{4}}.") == "\\frac{3}{4}"
    assert extract_boxed("no box") is None
    assert extract_boxed("\\boxed{unclosed {") is None
    assert extract_answer("x \\boxed{18}", "boxed") == "18"
    assert extract_answer("The answer is 7", "boxed") is None
    assert answers_match("\\dfrac{3}{4}", "\\frac{3}{4}")
    assert answers_match("90^\\circ", "90")
    tpl = "Q: {question}\nBox it: \\boxed{{}}\nA:\n"
    assert build_prompt(" hi ", tpl) == "Q: hi\nBox it: \\boxed{}\nA:\n"
