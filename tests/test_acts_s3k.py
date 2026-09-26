"""WP17-ACTS S3-K（results/acts/s3k/PROTOCOL.md）——先于实现。覆盖：坐标条件组合、池内 regret@3、兑现比例。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _g():
    return importlib.import_module("scripts.98_acts_s3k")


def test_coords():
    g = _g()
    a, ah = np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3])
    assert np.allclose(g.coords("Z", a, ah), [0, 0, 0])
    assert np.allclose(g.coords("P1", a, ah), [0.1, 0, 0])
    assert np.allclose(g.coords("T1+P23", a, ah), [1.0, 0.2, 0.3])
    assert np.allclose(g.coords("P1+T23", a, ah), [0.1, 2.0, 3.0])
    assert np.allclose(g.coords("T12", a, ah), [1.0, 2.0, 0])
    assert np.allclose(g.coords("T", a, ah), a) and np.allclose(g.coords("P", a, ah), ah)
    assert list(g.CONDS) == ["Z", "P", "P1", "T1", "T1+P23", "P1+T23", "T12", "T"]


def test_pool_regret():
    g = _g()
    wins = np.array([5.0, 4.0, 3.0, 0.0, 1.0])
    v0 = np.array([0.5, 0.7, 0.6, 0.9, 0.2]); v1 = np.array([0.55, 0.60, 0.65, 0.80, 0.30])
    # 前 3 = {0,1,2}，seed 0 选 1 → seed 1 0.60；oracle = seed 0 最好者 3 → seed 1 0.80
    assert np.isclose(g.pool_regret(wins, v0, v1, k=3), 0.20)


def test_realized():
    g = _g()
    assert np.isclose(g.realized(z=0.02, c=0.01, t=0.0, lower_better=True), 0.5)
    assert np.isclose(g.realized(z=0.2, c=0.3, t=0.4, lower_better=False), 0.5)
    assert np.isnan(g.realized(z=0.1, c=0.1, t=0.1, lower_better=True))
