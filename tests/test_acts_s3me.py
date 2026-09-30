"""WP17-ACTS S3-ME（results/acts/s3me/PROTOCOL.md）——先于实现。覆盖：相似度权重、混合预测、源级 E、超参选择平局规则、读数码、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _m():
    return importlib.import_module("scripts.114_acts_s3me")


def test_softmax_weights():
    m = _m()
    p = m.softmax_weights(np.array([0.9, 0.1, -0.5]), tau=0.1)
    assert abs(p.sum() - 1) < 1e-12 and p[0] > p[1] > p[2]
    q = m.softmax_weights(np.array([0.9, 0.1, -0.5]), tau=1e6)
    assert np.allclose(q, 1 / 3, atol=1e-5)                        # τ 很大 → 等权


def test_mixture_predict():
    m = _m()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 4)); b0 = rng.normal(size=4); c0 = 0.3
    W = rng.normal(size=(3, 4)); cw = np.array([0.1, -0.2, 0.0]); pi = np.array([0.5, 0.3, 0.2])
    s = m.mixture_predict(X, (b0, c0), (W, cw), pi, gamma=0.5)
    base = X @ b0 + c0
    assert np.allclose(s, base + 0.5 * (X @ (pi @ W) + pi @ cw))
    assert np.allclose(m.mixture_predict(X, (b0, c0), (W, cw), pi, gamma=0.0), base)


def test_source_E():
    m = _m()
    E = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]])
    groups = np.array(["a", "a", "b"])
    S = m.source_E(E, groups, ["b", "a"])
    assert np.allclose(S, [[0.0, 2.0], [2.0, 0.0]])


def test_select_grid():
    m = _m()
    sc = {(10, 0.1, 0.5): 0.30, (10, 0.5, 0.0): 0.30, (1, 0.2, 0.25): 0.29}
    assert m.select_grid(sc) == (10, 0.5, 0.0)                     # 平局 → 较小 γ
    sc2 = {(1, 0.1, 0.5): 0.30, (10, 0.5, 0.5): 0.30}
    assert m.select_grid(sc2) == (10, 0.5, 0.5)                    # 同 γ → 较大 τ


def test_verdict():
    m = _m()
    b = {"M1": 0.020, "M2": 0.30, "M4": 0.00}; u = {"M1": 0.021, "M2": 0.31, "M4": 0.02}
    assert m.verdict_s3me({"M1": 0.019, "M2": 0.32, "M4": 0.05}, u, b) == "S3ME-GO"
    assert m.verdict_s3me({"M1": 0.021, "M2": 0.32, "M4": 0.05}, u, b) == "S3ME-SAFE"
    assert m.verdict_s3me({"M1": 0.019, "M2": 0.305, "M4": 0.05}, u, b) == "S3ME-E-NULL"
    assert m.verdict_s3me({"M1": 0.030, "M2": 0.32, "M4": 0.05}, u, b) == "S3ME-HURT"


def test_cli():
    m = _m()
    a = m.build_parser().parse_args(["--run"])
    assert a.run
