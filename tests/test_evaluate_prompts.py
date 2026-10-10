import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<user>{messages[0]['content']}</user><assistant>"


def test_build_eval_prompts_preserves_plain_format():
    mod = load_module()
    rows = [{"question": "What is 2+2?"}]
    template = "Question: {question}\nAnswer:\n"
    assert mod.build_eval_prompts(rows, template) == ["Question: What is 2+2?\nAnswer:\n"]


def test_build_eval_prompts_chat_opens_assistant_turn_once():
    mod = load_module()
    rows = [{"question": "What is 2+2?"}]
    template = "Question: {question}\nShow work.\nAnswer:\n"
    prompts = mod.build_eval_prompts(rows, template, FakeTokenizer())
    assert prompts == ["<user>Question: What is 2+2?\nShow work.</user><assistant>"]
