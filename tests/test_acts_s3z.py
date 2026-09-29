"""WP17-ACTS S3-Z（results/acts/s3z/PROTOCOL.md）——先于实现。覆盖：成员数配置、搜索者数量曲线、选中者来源分类、读数码、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _z():
    return importlib.import_module("scripts.112_acts_s3z")


def test_configure_members():
    z = _z(); y = importlib.import_module("scripts.111_acts_s3y")
    z.configure()
    assert tuple(y.MEMBERS) == tuple(range(10)) and z.OLD_SEARCHERS == ("m0", "m1", "m2", "m3", "m4")


def test_searcher_curve():
    z = _z()
    picks = {"ens": ["a"], "m0": ["b"], "m1": ["c"]}
    s0 = {"a": 0.5, "b": 0.7, "c": 0.6}
    c = z.searcher_curve(picks, s0)
    assert abs(c[1] - np.mean([0.5, 0.7, 0.6])) < 1e-12                # 单个搜索者：各自最好的平均
    assert abs(c[2] - np.mean([0.7, 0.6, 0.7])) < 1e-12                # {a,b} {a,c} {b,c}
    assert c[3] == 0.7
    big = {f"m{i}": [f"k{i}"] for i in range(11)}
    sv = {f"k{i}": float(i) for i in range(11)}
    cb = z.searcher_curve(big, sv, max_combos=50, seed=0)
    assert set(cb) == set(range(1, 12)) and cb[11] == 10.0 and all(cb[m] <= cb[m + 1] + 1e-9 for m in range(1, 11))


def test_pick_origin():
    z = _z()
    assert z.pick_origin(["m2"]) == "old"
    assert z.pick_origin(["ens"]) == "new"                               # 10 成员集成是新的
    assert z.pick_origin(["m7"]) == "new"
    assert z.pick_origin(["m7", "m1"]) == "old"                          # S3-Y 成员也提名过 → 在 S3-Y 池里


def test_verdict():
    z = _z()
    assert z.verdict_s3z(0.650, 0.630) == "S3Z-BEATS-BOTH+UP+E"
    assert z.verdict_s3z(0.636, 0.634) == "S3Z-BEATS-TS2VEC+FLAT"
    assert z.verdict_s3z(0.620, 0.600) == "S3Z-BELOW+DOWN+E"


def test_cli():
    z = _z()
    a = z.build_parser().parse_args(["--search", "--arm", "NEW", "--dataset", "Cricket"])
    assert a.search and a.arm == "NEW"
    b = z.build_parser().parse_args(["--plan", "cv"])
    assert b.plan == "cv"
