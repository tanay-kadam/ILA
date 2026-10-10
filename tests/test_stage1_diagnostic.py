import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_stage1_outputs.py"
    spec = importlib.util.spec_from_file_location("diagnose_stage1_outputs", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_boxed_spans_handles_multiple_and_nested_boxes():
    mod = load_module()
    spans = mod.boxed_spans(r"First \boxed{2}, final \boxed{\frac{3}{4}}.")
    assert [s["content"] for s in spans] == ["2", r"\frac{3}{4}"]
    assert all(s["end"] > s["start"] for s in spans)


def test_boxed_spans_ignores_incomplete_final_box():
    mod = load_module()
    spans = mod.boxed_spans(r"Complete \boxed{7}; broken \boxed{8")
    assert [s["content"] for s in spans] == ["7"]


def test_loop_metrics_separates_strong_repetition():
    mod = load_module()
    repeated = " ".join(["therefore the answer repeats forever"] * 200)
    varied = " ".join(f"token{i}" for i in range(1000))
    rep = mod.loop_metrics(repeated)
    unique = mod.loop_metrics(varied)
    assert rep["likely_loop"]
    assert not unique["likely_loop"]
    assert rep["compression_ratio"] < unique["compression_ratio"]
    assert rep["unique_4gram_ratio"] < unique["unique_4gram_ratio"]


def test_audit_row_distinguishes_first_and_later_correct_boxes():
    mod = load_module()

    def position(text, end, total):
        return end

    row = {
        "uid": "x",
        "output": r"Maybe \boxed{4}. Correction: \boxed{5}." + " repeat" * 200,
        "n_gen_tokens": 500,
        "truncated": True,
        "correct": True,
        "gold": "5",
    }
    audited = mod.audit_row(row, position)
    assert audited["category"] == "truncated_later_box_correct"
    assert not audited["first_box_correct"]
    assert audited["first_correct_box_token"] > audited["first_box_token"]


def test_audit_row_identifies_correct_first_box_corrupted_by_later_loop():
    mod = load_module()

    def position(text, end, total):
        return end

    row = {
        "uid": "x",
        "output": r"Answer: \boxed{5}." + (r" loop \boxed{4}" * 200),
        "n_gen_tokens": 500,
        "truncated": True,
        "correct": False,
        "gold": "5",
    }
    audited = mod.audit_row(row, position)
    assert audited["category"] == "truncated_first_box_correct_then_corrupted"
    assert audited["first_box_correct"]
    assert not audited["last_box_correct"]
