from ila.choice_audit import map_boxed_choice

OPTIONS = ["A)40.8", "B)120", "C)0.5", "D)50%", "E)None of these"]


def test_unique_content_and_labels():
    assert map_boxed_choice("40.8", OPTIONS)["letter"] == "A"
    assert map_boxed_choice(r"\frac{1}{2}", OPTIONS)["letter"] == "C"
    assert map_boxed_choice(r"\text{None of these}", OPTIONS)["letter"] == "E"
    assert map_boxed_choice("B)120", OPTIONS)["letter"] == "B"
    assert map_boxed_choice(r"\mathrm{B}", OPTIONS)["kind"] == "strict_letter"


def test_conflicts_and_no_reasoning_search():
    assert map_boxed_choice("B)40.8", OPTIONS)["kind"] == "ambiguous_label_content"
    assert map_boxed_choice("The answer is 40.8", OPTIONS)["letter"] is None
    assert map_boxed_choice("I computed 40.8 so A", OPTIONS)["letter"] is None
    assert map_boxed_choice(None, OPTIONS)["letter"] is None


def test_units_and_percent_are_preserved():
    assert map_boxed_choice("50", OPTIONS)["letter"] is None
    assert map_boxed_choice("50%", OPTIONS)["letter"] == "D"
    assert map_boxed_choice("40.8 km", OPTIONS)["letter"] is None


def test_duplicates_are_not_disambiguated_using_gold():
    options = ["A)0.5", "B)1/2", "C)2", "D)3", "E)4"]
    assert map_boxed_choice("0.5", options)["kind"] == "ambiguous_duplicate_content"
    assert map_boxed_choice("A)0.5", options)["letter"] is None
    assert map_boxed_choice("A", options)["letter"] == "A"
