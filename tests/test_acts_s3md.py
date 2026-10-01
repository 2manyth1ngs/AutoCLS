"""WP17-ACTS S3-MD（results/acts/s3md/PROTOCOL.md）——先于实现。覆盖：任务内标准化、源专家残差拟合、成员分数组合、
靶的源权重、读数码、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _d():
    return importlib.import_module("scripts.115_acts_s3md")


def test_task_z():
    d = _d()
    z = d.task_z(np.array([1.0, 2.0, 3.0]))
    assert abs(z.mean()) < 1e-12 and abs(z.std() - 1) < 1e-9
    assert np.all(d.task_z(np.array([2.0, 2.0])) == 0)


def test_fit_experts():
    d = _d(); m = importlib.import_module("scripts.114_acts_s3me")
    rng = np.random.default_rng(0)
    X = [rng.normal(size=(6, 3)) for _ in range(4)]
    Z = [rng.normal(size=6) for _ in range(4)]; S0 = [rng.normal(size=6) for _ in range(4)]
    groups = np.array(["a", "a", "b", "b"])
    W, c = d.fit_experts(X, Z, S0, groups, ["b", "a"], alpha=10.0, idx=[0, 1, 2, 3])
    assert W.shape == (2, 3) and c.shape == (2,)
    Xb = np.vstack([X[2], X[3]]); Rb = np.concatenate([Z[2] - d.task_z(S0[2]), Z[3] - d.task_z(S0[3])])
    wb, cb = m.ridge_fit(Xb, Rb, 10.0)
    assert np.allclose(W[0], wb) and abs(c[0] - cb) < 1e-12            # 源 b = 自己切片残差的岭回归


def test_member_score():
    d = _d()
    rng = np.random.default_rng(1)
    zs0 = rng.normal(size=5); X = rng.normal(size=(5, 3)); W = rng.normal(size=(2, 3)); c = np.array([0.1, -0.3]); pi = np.array([0.7, 0.3])
    s = d.member_score(zs0, X, W, c, pi, 0.5)
    assert np.allclose(s, zs0 + 0.5 * (X @ (pi @ W) + pi @ c))
    assert np.allclose(d.member_score(zs0, X, W, c, pi, 0.0), zs0)


def test_target_pi():
    d = _d()
    Es = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]); groups = np.array(["a", "a", "b"])
    pi = d.target_pi(np.array([1.0, 0.0]), Es, groups, ["a", "b"], tau=0.1, uniform=False)
    assert pi[0] > 0.99 and abs(pi.sum() - 1) < 1e-12
    assert np.allclose(d.target_pi(np.array([1.0, 0.0]), Es, groups, ["a", "b"], tau=0.1, uniform=True), [0.5, 0.5])


def test_verdict():
    d = _d()
    assert d.verdict_s3md(0.650, 0.640, 0.6377) == "S3MD-BEATS-BOTH+E+EXP"
    assert d.verdict_s3md(0.630, 0.635, 0.6377) == "S3MD-BEATS-TS2VEC"
    assert d.verdict_s3md(0.620, 0.610, 0.600) == "S3MD-BELOW+E+EXP"


def test_cli():
    d = _d()
    a = d.build_parser().parse_args(["--search", "--arm", "ME", "--dataset", "Cricket"])
    assert a.search and a.arm == "ME"
    assert set(d.ARMS_MD) == {"ME", "UNI"}
