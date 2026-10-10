import pytest
from ila.gpu_lease import gpu_lease


def test_forbidden_gpus_never_acquire_a_lease(tmp_path):
    for gpu in (5,6):
        with pytest.raises(ValueError):
            with gpu_lease(tmp_path,gpu):pass


def test_cross_branch_lease_prevents_double_allocation_and_releases(tmp_path):
    with gpu_lease(tmp_path,1) as first:
        assert first
        with gpu_lease(tmp_path,1) as second:
            assert not second
    with gpu_lease(tmp_path,1) as third:
        assert third
