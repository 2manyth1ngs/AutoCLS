"""WP17-ACTS UT-ENS（results/acts/utens/PROTOCOL.md）——先于实现。覆盖：CV 排序（第 1 名 = select_cv）、逐块 L2 拼接、LPT 分配、读数码、作业表、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _e():
    return importlib.import_module("scripts.119_acts_utens")


def test_rank_cv_matches_select_cv():
    e = _e(); y = importlib.import_module("scripts.111_acts_s3y")
    keys = ["a", "b", "c", "d", "e"]
    cv = {"a": [0.5, 0.6], "b": [0.7, 0.7, 0.7], "c": [0.7, 0.7, 0.7], "d": [0.2], "e": []}
    ens = {"a": 0.0, "b": 0.1, "c": 0.3, "d": 9.0}
    r = e.rank_cv(keys, cv, ens)
    assert r[0] == y.select_cv(keys, cv, ens) == "c"                   # 平局取集成分数高者
    assert r[:3] == ["c", "b", "a"] and "e" not in r                    # 无 CV 分的不进排序


def test_concat_l2():
    e = _e()
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(4, 3)), 10 * rng.normal(size=(4, 2))
    Z = e.concat_l2([A, B])
    assert Z.shape == (4, 5)
    assert np.allclose(np.linalg.norm(Z[:, :3], axis=1), 1) and np.allclose(np.linalg.norm(Z[:, 3:], axis=1), 1)


def test_lpt_assign():
    e = _e()
    costs = {**{f"h{i}": 10.0 for i in range(4)}, **{f"l{i}": 1.0 for i in range(4)}}   # py3.8（服务器）无 dict |
    parts = e.lpt_assign(costs, 4)
    assert len(parts) == 4 and sorted(j for p in parts for j in p) == sorted(costs)
    assert all(sum(j.startswith("h") for j in p) == 1 for p in parts)  # 重作业不扎堆


def test_verdict():
    e = _e()
    assert e.verdict_ens(0.640, 0.630, 0.620) == "ENS-BEATS-BOTH+GAIN+E+TA"
    assert e.verdict_ens(0.590, 0.600, 0.580) == "ENS-BEATS-TS2VEC+GAIN+TA"
    assert e.verdict_ens(0.570, 0.560, 0.600) == "ENS-BELOW+E"


def test_jobs_and_cli():
    e = _e()
    jobs = e.job_list()
    assert len(jobs) == 33 and ("FaceDetection", 2) in jobs
    a = e.build_parser().parse_args(["--run", "--worker", "1", "--workers", "4"])
    assert a.run and a.worker == 1 and a.workers == 4 and e.K_ENS == 3
