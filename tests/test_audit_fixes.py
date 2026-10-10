import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ila.dose_gate import contrast_gate, outcome


def signal():
    return {"ci95_hierarchical": [-20, -10], "rel_delta_tokens": -.10, "sign_consistent_across_seeds": True}


def test_significant_degraded_result_is_quality_failure_not_null():
    gate = contrast_gate(signal(), -.1, .35)
    assert gate["signal_gate_pass"]
    assert not gate["guardrails_pass"]
    assert outcome([gate]) == "STOP_QUALITY_FAILURE"


def test_two_collapsed_models_cannot_pass_on_relative_accuracy_alone():
    gate = contrast_gate(signal(), -.1, .001, baseline_pass=False)
    assert not gate["continue_to_step230"]
    assert outcome([gate]) == "STOP_QUALITY_FAILURE"


def test_stable_signal_and_stable_null_are_distinct():
    assert outcome([contrast_gate(signal(), 0, .001)]) == "STABLE_SIGNAL_REVIEW"
    null = {"ci95_hierarchical": [-2, 2], "rel_delta_tokens": .001, "sign_consistent_across_seeds": False}
    assert outcome([contrast_gate(null, 0, .001)]) == "STOP_NO_SIGNAL"


def test_truncation_boundary_is_strict():
    assert not contrast_gate(signal(), 0, .02)["guardrails_pass"]
    assert contrast_gate(signal(), .01, .019)["guardrails_pass"]


def evaluate_module():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate.py"
    spec = importlib.util.spec_from_file_location("audit_evaluate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hf_loads_the_recorded_adapter_before_device_and_eval(monkeypatch):
    mod = evaluate_module()
    calls = []
    class Model:
        def to(self, device):
            calls.append(("device", device)); return self
        def eval(self):
            calls.append(("eval",)); return self
    model = Model()
    def load_base(name, dtype):
        calls.append(("base", name)); return model
    def load_adapter(base, path):
        assert base is model
        calls.append(("adapter", path)); return base
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_base)))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=load_adapter)))
    assert mod.load_hf_model("base", "bf16", "cuda", "adapter") is model
    assert calls == [("base", "base"), ("adapter", "adapter"), ("device", "cuda"), ("eval",)]


def test_cache_rejects_legacy_and_changed_provenance():
    mod = evaluate_module()
    with pytest.raises(SystemExit):
        mod.check_cached_summary({}, "new")
    with pytest.raises(SystemExit):
        mod.check_cached_summary({"evaluation_fingerprint": "old"}, "new")
    mod.check_cached_summary({"evaluation_fingerprint": "new"}, "new")


def test_cache_fingerprint_changes_with_data_adapter_or_config(tmp_path):
    mod = evaluate_module()
    data = tmp_path / "data.jsonl"; data.write_text("original")
    adapter = tmp_path / "adapter"; adapter.mkdir()
    weights = adapter / "adapter_model.safetensors"; weights.write_bytes(b"weights")
    record = {"model_dir": "base", "adapter_dir": str(adapter)}
    def fingerprint(cfg):
        return mod.evaluation_fingerprint(cfg, record, data, "high", 0, None, True)
    original = fingerprint({"lr": 1e-5})
    assert original != fingerprint({"lr": 3e-6})
    data.write_text("changed")
    assert original != fingerprint({"lr": 1e-5})
    data.write_text("original"); weights.write_bytes(b"new weights")
    assert original != fingerprint({"lr": 1e-5})
