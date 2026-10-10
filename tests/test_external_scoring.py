from ila.textio import answers_match_choice, extract_answer


def test_explicit_choice_normalization():
    for prediction in ("B", "b", "(B)", "\\text{B}", "\\mathrm{B}"):
        assert answers_match_choice(prediction, "B")
    for prediction in (None, "A", "B or C", "25", "The answer is B"):
        assert not answers_match_choice(prediction, "B")


def test_choice_uses_last_complete_box_under_existing_extraction_rule():
    assert extract_answer("\\boxed{A} then \\boxed{B}", "boxed_choice") == "B"
