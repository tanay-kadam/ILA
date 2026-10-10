import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_stage1 as a1  # noqa: E402

ACFG = {"f_test_alpha": 0.05, "min_cost_range_rel": 0.05, "heuristic_share_stop": 0.8, "heuristic_share_go": 0.6}


def test_pooled_noise_var():
    v, df = a1.pooled_noise_var([np.array([1.0, 3.0, 5.0]), np.array([0.0, 2.0])])
    assert df == 3
    assert v == pytest.approx((2 * 4.0 + 1 * 2.0) / 3)


def test_ridge_loo_matches_brute_force():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.1, size=12)
    press, _ = a1.ridge_press(X, y[None, :], [0.5])
    brute = 0.0
    for i in range(12):
        m = np.arange(12) != i
        Xa = np.column_stack([np.ones(11), X[m]])
        P = 0.5 * np.eye(4)
        P[0, 0] = 0
        w = np.linalg.solve(Xa.T @ Xa + P, Xa.T @ y[m])
        brute += (y[i] - np.r_[1.0, X[i]] @ w) ** 2
    assert press[0, 0] == pytest.approx(brute)


def test_heuristic_r2_high_when_linear_low_when_noise():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(16, 2))
    r2_lin, _ = a1.heuristic_r2(X, (3 * X[:, 0] + 0.05 * rng.normal(size=16))[None, :], [0.1, 1.0])
    r2_noise, _ = a1.heuristic_r2(X, rng.normal(size=(1, 16)), [0.1, 1.0])
    assert r2_lin[0] > 0.9
    assert r2_noise[0] < 0.3


def test_decide_rules():
    assert a1.decide(0.5, 0.2, 0.1, ACFG)[0] == "STOP_NO_EFFECT"
    assert a1.decide(0.001, 0.02, 0.1, ACFG)[0] == "STOP_NO_EFFECT"
    assert a1.decide(0.001, 0.2, 0.9, ACFG)[0] == "STOP_HEURISTICS_SUFFICE"
    assert a1.decide(0.001, 0.2, 0.3, ACFG)[0] == "GO_STAGE2"
    assert a1.decide(0.001, 0.2, 0.7, ACFG)[0] == "AMBIGUOUS"


def test_stage1b_pilot_gate_requires_both_sets_noninferior_and_low_truncation():
    pc = {"max_truncation_rate": 0.2, "max_accuracy_drop_abs": 0.05}
    passing = {
        "gsm8k_test": {"accuracy": 0.87, "base_accuracy": 0.91, "truncation_rate": 0.01},
        "math500": {"accuracy": 0.68, "base_accuracy": 0.722, "truncation_rate": 0.02},
    }
    result = a1.pilot_gate(passing, pc)
    assert result["verdict"] == "PASS"
    assert result["accuracy_ok"] == {"gsm8k_test": True, "math500": True}

    failing = {k: dict(v) for k, v in passing.items()}
    failing["math500"]["accuracy"] = 0.67
    assert a1.pilot_gate(failing, pc)["verdict"] == "FAIL"

    failing = {k: dict(v) for k, v in passing.items()}
    failing["gsm8k_test"]["truncation_rate"] = 0.2
    assert a1.pilot_gate(failing, pc)["verdict"] == "FAIL"


def _fake_project(tmp: Path, scenario: str, rng: np.random.Generator) -> Path:
    data, res = tmp / "data", tmp / "results"
    n_q = 40
    b_sets = [f"b{j:02d}" for j in range(16)]
    arms = ["armA_high", "armA_low", "armA_random"]
    refl = {n: float(rng.uniform(1, 5)) for n in b_sets + arms}
    hidden = {n: float(rng.normal()) for n in b_sets + arms}
    for n in b_sets + arms:
        rows = [{"uid": f"u{i}", "question": "q", "target": "t", "source": f"s{i % 3}", "n_tokens": 2000 + 10 * j,
                 "reflection_density": refl[n] + rng.normal(scale=0.01), "restart_count": 1, "boxed_count": 1}
                for i, j in zip(range(30), range(30))]
        (data).mkdir(parents=True, exist_ok=True)
        (data / f"train_{n}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    base_q = rng.uniform(800, 1200, size=2 * n_q)

    def effect(n):
        return {"heuristic": 60.0 * (refl[n] - 3), "hidden": 60.0 * hidden[n], "none": 0.0}[scenario]

    def write_run(n, s):
        for k, es in enumerate(("gsm8k_test", "math500")):
            d = res / n / f"seed{s}" / "eval" / es
            d.mkdir(parents=True, exist_ok=True)
            toks = base_q[k * n_q:(k + 1) * n_q] + effect(n) + rng.normal(scale=8, size=n_q)
            outs = [{"uid": f"{es}{i}", "n_gen_tokens": int(t), "correct": bool(i % 2), "truncated": False}
                    for i, t in enumerate(toks)]
            (d / "outputs.jsonl").write_text("\n".join(json.dumps(o) for o in outs))
            (d / "summary.json").write_text(json.dumps({"mean_tokens": float(toks.mean()), "accuracy": 0.5,
                                                         "truncation_rate": 0.0}))

    for n in arms:
        for s in (0, 1, 2):
            write_run(n, s)
    for n in b_sets:
        write_run(n, 0)
    for n in ("b00", "b01"):
        for s in (1, 2):
            write_run(n, s)
    cfg = yaml.safe_load(Path(__file__).resolve().parents[1].joinpath("configs/stage1.yaml").read_text())
    cfg["data_dir"], cfg["results_dir"] = str(data), str(res)
    cfg["analysis"]["n_boot"] = 200
    p = tmp / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


@pytest.mark.parametrize("scenario,expected", [
    ("heuristic", "STOP_HEURISTICS_SUFFICE"),
    ("hidden", "GO_STAGE2"),
    ("none", "STOP_NO_EFFECT"),
])
def test_end_to_end_scenarios(tmp_path, monkeypatch, scenario, expected):
    cfgp = _fake_project(tmp_path, scenario, np.random.default_rng(7))
    monkeypatch.setattr(sys, "argv", ["analyze_stage1.py", "--config", str(cfgp)])
    a1.main()
    rep = json.loads((tmp_path / "results" / "analysis.json").read_text())
    assert rep["missing_runs"] == []
    assert rep["verdict"]["decision"] == expected, rep["armB"]
