"""WP17-ACTS UT（results/acts/ut/PROTOCOL.md）——先于实现。覆盖：未见靶规则、成员映射、基线数字、读数码、部署靶直读路径、命令行。"""
from __future__ import annotations

import importlib

import numpy as np


def _u():
    return importlib.import_module("scripts.118_acts_ut")


def test_unseen_targets_rule():
    u = _u()
    rows = {"A": {"length_max": "100", "equal_length": "True"}, "B": {"length_max": "30", "equal_length": "True"},
            "C": {"length_max": "100", "equal_length": "False"}, "D": {"length_max": "17984", "equal_length": "True"},
            "S": {"length_max": "100", "equal_length": "True"}, "InsectWingbeat": {"length_max": "100", "equal_length": "True"}}
    assert u.unseen_targets(rows, sources={"S"}) == ["A"]
    assert u.UT_TARGETS == ("AtrialFibrillation", "BasicMotions", "DuckDuckGeese", "Epilepsy", "FaceDetection", "FingerMovements",
                            "HandMovementDirection", "Heartbeat", "MotorImagery", "SelfRegulationSCP2", "StandWalkJump")


def test_member_fold():
    u = _u()
    assert [u.member_fold(q) for q in range(8)] == ["Handwriting", "PhonemeSpectra", "UWaveGestureLibrary", "Cricket",
                                                    "EthanolConcentration", "SelfRegulationSCP1", "Handwriting", "PhonemeSpectra"]


def test_baselines():
    u = _u()
    assert set(u.TS2VEC_UT) == set(u.UT_TARGETS) == set(u.INFOTS_S_UT)
    assert abs(np.mean(list(u.TS2VEC_UT.values())) - 0.5796) < 1e-4 and abs(np.mean(list(u.INFOTS_S_UT.values())) - 0.6308) < 1e-4


def test_verdict():
    u = _u()
    assert u.verdict_ut(0.640, 0.630, 0.620) == "UT-BEATS-BOTH+E+EXP+TA"
    assert u.verdict_ut(0.600, 0.610, 0.605) == "UT-BEATS-TS2VEC+EXP"
    assert u.verdict_ut(0.570, 0.560, 0.575) == "UT-BELOW+E"


def test_load_target_full_matches_source_loader():
    ac = importlib.import_module("scripts.acts_common")
    X1, y1 = ac.load_target_full("ERing"); X2, y2 = ac.load_dataset_full("ERing")
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)


def test_whole_loader_uses_target_path(monkeypatch):
    ac = importlib.import_module("scripts.acts_common"); g83 = importlib.import_module("scripts.83_acts_corpus")
    calls = []
    X = np.random.default_rng(0).normal(size=(20, 60, 2)).astype(np.float32); y = np.array([0, 1] * 10)
    monkeypatch.setattr(ac, "load_target_full", lambda n: (calls.append(("target", n)) or (X, y)))
    monkeypatch.setattr(ac, "load_dataset_full", lambda n: (calls.append(("source", n)) or (X, y)))
    cfg = ac.load_acts_config(profile="s1pp")
    tm = {"Z_whole_official": {"dataset": "Z", "eval_mode": "official", "n_train_official": 10, "deploy_target": True},
          "Q_whole_official": {"dataset": "Q", "eval_mode": "official", "n_train_official": 10}}
    ld = g83.make_whole_loader(cfg, tm)
    ld("Z_whole_official"); ld("Q_whole_official")
    assert calls == [("target", "Z"), ("source", "Q")]


def test_cli():
    u = _u()
    assert u.build_parser().parse_args(["--search", "--arm", "BLIND", "--dataset", "Epilepsy"]).arm == "BLIND"


def test_eval_timeout_env_override(monkeypatch):
    g39 = importlib.import_module("scripts.39_sh_deploy")
    monkeypatch.setenv("AUTOCLS_EVAL_TIMEOUT_S", "1800")
    assert importlib.reload(g39).EVAL_TIMEOUT_S == 1800
    monkeypatch.delenv("AUTOCLS_EVAL_TIMEOUT_S")
    assert importlib.reload(g39).EVAL_TIMEOUT_S == 480            # 默认不变
