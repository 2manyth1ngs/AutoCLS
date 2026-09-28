"""WP17-ACTS S3-X（results/acts/s3x/PROTOCOL.md）——先于实现。覆盖：候选池去重、已跑格复用、池内 regret 与随机期望、
诚实池内最好、自我偏好、读数码、命令行。零 GPU。"""
from __future__ import annotations

import importlib
from math import comb

import numpy as np


def _x():
    return importlib.import_module("scripts.110_acts_s3x")


def test_build_pool():
    x = _x()
    s = {"FULL": {"D1": [{"cls_key": "a", "cls": {}}, {"cls_key": "b", "cls": {}}]},
         "A": {"D1": [{"cls_key": "b", "cls": {}}, {"cls_key": "c", "cls": {}}]}}
    p = x.build_pool(s, ["FULL", "A"], ["D1"])
    assert [(e["cls_key"], e["arms"]) for e in p["D1"]] == [("a", ["FULL"]), ("b", ["FULL", "A"]), ("c", ["A"])]


def test_import_rows():
    x = _x()
    cells = [{"key": "k1", "task_id": "T_whole_official", "cls_key": "a", "seed": 0, "eval": "official"},
             {"key": "k2", "task_id": "T_whole_official", "cls_key": "b", "seed": 0, "eval": "official"},
             {"key": "k3", "task_id": "T_whole_official", "cls_key": "b", "seed": 1, "eval": "official"}]
    old = [{"key": "k1", "task_id": "T_whole_official", "cls_key": "a", "seed": 0, "eval": "official", "acc": 0.5, "error": None},
           {"key": "zz", "task_id": "T_whole_official", "cls_key": "b", "seed": 0, "eval": "official", "acc": 0.6, "error": None},   # 键不同 → 按四元组对上
           {"key": "yy", "task_id": "T_whole_official", "cls_key": "b", "seed": 1, "eval": "official", "acc": 0.7, "error": "boom"}]  # 出错不复用
    rows, n_direct, n_loose = x.import_rows(cells, old)
    assert n_direct == 1 and n_loose == 1
    assert {r["key"]: r["acc"] for r in rows} == {"k1": 0.5, "k2": 0.6}
    assert all(r["key"] in ("k1", "k2") for r in rows)


def test_pool_regret_and_random():
    x = _x()
    truth = np.array([0.5, 0.9, 0.7, 0.8, 0.6])
    assert abs(x.pool_regret(np.array([5, 1, 4, 3, 2]), truth, k=3) - 0.1) < 1e-12      # 前 3 = {0, 2, 3} → 最好 0.8
    # 随机 3 条的期望最大值：P(max = v_(j)) = C(j−1, 2) / C(5, 3)
    v = np.sort(truth); e = sum(comb(j - 1, 2) / comb(5, 3) * v[j - 1] for j in range(3, 6))
    assert abs(x.random_k_regret(truth, 3) - (0.9 - e)) < 1e-12
    assert x.random_k_regret(truth, 5) == 0.0


def test_honest_best():
    x = _x()
    s0 = np.array([0.6, 0.5, 0.7]); s1 = np.array([0.6, 0.9, 0.4]); s2 = np.array([0.55, 0.65, 0.8])
    assert x.honest_best(s0, s1, s2) == 0.65                                               # 按 (s0 + s1) / 2 选 1 号，报 s2


def test_self_pref():
    x = _x()
    assert x.self_pref(np.array([9, 1, 8, 7, 2]), own=[0, 1, 2], k=3) == 2


def test_verdict():
    x = _x()
    assert x.verdict_s3x(0.63, 0.010, 0.040) == "S3X-FIND-OK+RANK-OK"
    assert x.verdict_s3x(0.61, 0.035, 0.040) == "S3X-FIND-PARTIAL+RANK-FAIL"
    assert x.verdict_s3x(0.59, 0.025, 0.040) == "S3X-FIND-FAIL+RANK-WEAK"


def test_cli():
    x = _x()
    a = x.build_parser().parse_args(["--score", "--arm", "FULL", "--device", "cpu"])
    assert a.score and a.arm == "FULL"
    assert set(x.RANKERS) == {"FULL", "WO_SETTF", "WO_TS2VEC", "WO_SHARED", "BLIND", "A", "D"}
