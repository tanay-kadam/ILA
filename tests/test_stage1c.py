import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_adapter_snapshot_callback_saves_only_requested_steps(tmp_path):
    mod = load_script("train_sft.py")

    class Model:
        def save_pretrained(self, path):
            (path / "adapter_config.json").write_text("{}")

    cb = mod.AdapterSnapshotCallback(tmp_path, [2, 5])
    control = object()
    for step in [1, 2, 2, 3, 5]:
        returned = cb.on_step_end(
            None,
            SimpleNamespace(global_step=step, is_world_process_zero=True),
            control,
            model=Model(),
        )
        assert returned is control
    manifest = json.loads((tmp_path / "trajectory_adapters" / "manifest.json").read_text())
    assert manifest["saved_steps"] == [2, 5]
    assert (tmp_path / "trajectory_adapters" / "step002" / "adapter_config.json").is_file()


def test_stage1c_split_order_is_seeded_and_stable():
    mod = load_script("materialize_stage1c_splits.py")
    rows = [{"uid": str(i)} for i in range(20)]
    assert mod.ordered(rows, 7) == mod.ordered(list(reversed(rows)), 7)
    assert mod.ordered(rows, 7) != mod.ordered(rows, 8)


def test_stop_after_steps_preserves_external_schedule_and_stops_at_boundary():
    mod = load_script("train_sft.py")
    cb = mod.StopAfterStepsCallback(15)
    control = SimpleNamespace(should_training_stop=False)
    cb.on_step_end(None, SimpleNamespace(global_step=14), control)
    assert not control.should_training_stop
    cb.on_step_end(None, SimpleNamespace(global_step=15), control)
    assert control.should_training_stop
