import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_sft.py"
    spec = importlib.util.spec_from_file_location("train_sft", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class FakeTokenizer:
    eos_token_id = 99

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [10, 11] if "Question" in text else [20, 21]}

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is True
        if len(messages) == 1:
            assert add_generation_prompt is True
            assert messages[0]["content"].endswith("Show work.")
            return [1, 2, 3]
        assert add_generation_prompt is False
        assert messages[1] == {"role": "assistant", "content": "solution"}
        return [1, 2, 3, 7, 99]


class MappingTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        ids = super().apply_chat_template(messages, tokenize, add_generation_prompt)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_chat_dataset_masks_user_and_trains_on_assistant_turn():
    mod = load_module()
    rows = [{"question": "What?", "target": "solution"}]
    ds = mod.SFTDataset(
        rows,
        FakeTokenizer(),
        max_len=20,
        template="Question: {question}\nShow work.\nAnswer:\n",
        canonical_chat=True,
    )
    assert ds[0]["input_ids"] == [1, 2, 3, 7, 99]
    assert ds[0]["labels"] == [-100, -100, -100, 7, 99]
    assert ds.n_truncated == 0


def test_plain_dataset_behavior_is_unchanged():
    mod = load_module()
    rows = [{"question": "What?", "target": "solution"}]
    ds = mod.SFTDataset(rows, FakeTokenizer(), max_len=20, canonical_chat=False)
    assert ds[0]["input_ids"] == [10, 11, 20, 21, 99]
    assert ds[0]["labels"] == [-100, -100, 20, 21, 99]


def test_chat_dataset_accepts_transformers5_mapping_result():
    mod = load_module()
    rows = [{"question": "What?", "target": "solution"}]
    ds = mod.SFTDataset(
        rows,
        MappingTokenizer(),
        max_len=20,
        template="Question: {question}\nShow work.\nAnswer:\n",
        canonical_chat=True,
    )
    assert ds[0]["input_ids"] == [1, 2, 3, 7, 99]
    assert ds[0]["labels"] == [-100, -100, -100, 7, 99]
