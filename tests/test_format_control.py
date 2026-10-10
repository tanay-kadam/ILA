import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tmlr_format_control import control_template
from ila.textio import build_chat_user_content


def test_template_preserves_question_reasoning_and_braces():
    original = r"Question: {question}" + "\n" + r"Please reason step by step, and put your final answer within \boxed{{}}." + "\nAnswer:\n"
    template = control_template(original)
    content = build_chat_user_content("UNIQUEQUESTION A)10 B)20 C)30 D)40 E)50", template)
    assert content.count("UNIQUEQUESTION") == 1
    assert r"\boxed{A}" in content and r"\boxed{D}" in content
    assert "reason step by step" in content
    assert "exactly one option letter" in content
    assert "Answer:" not in content
    assert "answer-only" not in content
