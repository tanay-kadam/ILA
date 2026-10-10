import numpy as np
import pytest
from ila.cost_boundary import measures, decompose, bootstrap


def test_decomposition_is_exact_when_frequency_and_finished_length_change():
    a = measures([6144,6144,400,800], [True,True,False,False])
    b = measures([6144,200,500,900], [True,False,False,False])
    d = decompose(a,b)
    assert d["frequency_component"] + d["finished_length_component"] == pytest.approx(d["delta_tokens"])
    assert d["capped_row_difference"] + d["finished_row_difference"] == pytest.approx(d["delta_tokens"])


def test_all_capped_does_not_invent_a_finished_mean():
    a = measures([6144,6144],[True,True]); b=measures([200,400],[False,False])
    assert a["finished_mean_tokens"] is None
    assert not decompose(a,b)["symmetric_identified"]


def test_invalid_cap_rows_fail():
    with pytest.raises(ValueError): measures([1024],[True])


def test_paired_identical_arms_have_exact_zero_bootstrap():
    a={"tokens":np.array([[100,6144],[120,6144],[130,6144]]), "truncated":np.array([[False,True]]*3), "correct":np.array([[True,False]]*3)}
    result=bootstrap(a,a,draws=40,paired=True)
    assert result["delta_tokens"]["ci95"] == [0.,0.]
