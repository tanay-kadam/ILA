import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import tmlr_eos_control as controller


def test_waiting_worker_exits_after_all_jobs_claimed(tmp_path,monkeypatch):
    (tmp_path/"jobs.json").write_text(json.dumps([{"id":"already_claimed"}]))
    (tmp_path/"claims/already_claimed").mkdir(parents=True)
    monkeypatch.setattr(controller,"CONTROL",tmp_path)
    monkeypatch.setattr(controller,"ready",lambda gpu:False)
    monkeypatch.setattr(controller.time,"sleep",lambda _:pytest.fail("must not wait after all claims exist"))
    assert controller.acquire(0)==(None,None)


@pytest.mark.parametrize("gpu",[5,6])
def test_forbidden_worker_rejected_before_data_or_device_access(gpu):
    with pytest.raises(ValueError,match="forbidden GPU"):
        controller.worker(gpu)
