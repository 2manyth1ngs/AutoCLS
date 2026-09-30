"""WP17-ACTS S3-Z′（results/acts/s3zp/PROTOCOL.md）——先于实现。覆盖：决赛名单、两阶段选 1、重复折任务 id 与折划分、读数码、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _p():
    return importlib.import_module("scripts.113_acts_s3zp")


def test_finalists():
    p = _p()
    cv1 = {"a": 0.80, "b": 0.90, "c": 0.85, "d": 0.85, "e": 0.70}
    ens = {"a": 0.0, "b": 0.0, "c": 1.0, "d": 2.0, "e": 9.0}
    assert p.finalists(["a", "b", "c", "d", "e"], cv1, ens, k=3) == ["b", "d", "c"]      # 平局取集成分数高者
    assert p.finalists(["a", "b"], cv1, ens, k=4) == ["b", "a"]


def test_two_stage_pick():
    p = _p()
    fin = ["b", "d", "c"]
    cv9 = {"b": [0.9] * 3 + [0.7] * 6, "d": [0.8] * 9, "c": [0.85] * 9}
    cv1 = {"b": 0.90, "d": 0.85, "c": 0.85}; ens = {"b": 0.0, "d": 2.0, "c": 1.0}
    assert p.two_stage_pick(fin, cv9, cv1, ens) == "c"                                    # 9 次均值最高
    cv9t = {"b": [0.8] * 9, "d": [0.8] * 9, "c": [0.7] * 9}
    assert p.two_stage_pick(fin, cv9t, cv1, ens) == "b"                                   # 平局 → 第一阶段均值高者
    cv9t2 = {"b": [0.7] * 9, "d": [0.8] * 9, "c": [0.8] * 9}
    assert p.two_stage_pick(fin, cv9t2, {"b": 0.9, "d": 0.85, "c": 0.85}, ens) == "d"     # 再平局 → 集成分数高者


def test_rep_tasks():
    p = _p()
    ac = importlib.import_module("scripts.acts_common")
    assert p.rep_tid("Cricket", 2, 1) == "Cricket_whole_tcv2_r1"
    y = np.repeat(np.arange(4), 9)
    s0, s1, s2 = (p.rep_fold_seed("Cricket", r) for r in (0, 1, 2))
    assert len({s0, s1, s2}) == 3
    f0, f1 = ac.tcv_folds(y, 3, s0), ac.tcv_folds(y, 3, s1)
    assert not np.array_equal(f0, f1)                                                    # 重复用不同折划分
    for f in (f0, f1):
        assert all(sorted(np.bincount(f[y == c], minlength=3).tolist()) == [3, 3, 3] for c in range(4))   # 分层


def test_verdict():
    p = _p()
    assert p.verdict_s3zp(0.650, 0.630) == "S3ZP-BEATS-BOTH+UP+E"
    assert p.verdict_s3zp(0.630, 0.635) == "S3ZP-BEATS-TS2VEC+FLAT"
    assert p.verdict_s3zp(0.615, 0.600) == "S3ZP-BELOW+DOWN+E"


def test_cli():
    p = _p()
    a = p.build_parser().parse_args(["--prep"])
    assert a.prep and not a.analyze
