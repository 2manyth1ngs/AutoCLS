"""WP17-ACTS S3-L（results/acts/s3l/PROTOCOL.md）——先于实现。覆盖：线性打分 → 胜场、幅度校准斜率与截断、选择规则、读数码、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _g():
    return importlib.import_module("scripts.99_acts_s3l")


def test_wins_from_linear():
    g = _g()
    H = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
    assert g.wins_linear(H, np.zeros(3)).tolist() == [0, 2, 1]                        # 只看 s₀
    assert g.wins_linear(H, np.array([2.0, 0, 0])).tolist() == [2, 0, 1]              # s₀ + 2·g₁：0→2、1→1、2→1.5


def test_calib_slope():
    g = _g()
    rng = np.random.default_rng(0)
    a = rng.normal(size=200); ah = 0.1 * a + 0.01 * rng.normal(size=200)              # 方向对、幅度被压 10 倍
    b = g.calib_slope(a, ah)
    assert 5.0 < b <= a.std() / ah.std() + 1e-9                                       # 放大，但不超过真实幅度
    assert g.calib_slope(a, -ah) == 0.0                                               # 反向 → 0
    assert g.calib_slope(a, np.zeros(200)) == 0.0


def test_select():
    g = _g()
    crit = {("sh4", "MSE"): 0.30, ("sh4", "DS"): 0.34, ("pt4", "DS"): 0.34, ("pt4", "MSE"): 0.1}
    assert g.select_best(crit) == ("pt4", "DS")                                         # 平局取字典序最小的 (E 来源, 映射)
    assert g.select_best({("sh4", "MSE"): 0.5, ("sh8", "DS"): 0.4}) == ("sh4", "MSE")


def _c(d, p, npos):
    return {"d": d, "p": p, "npos": npos}


def test_verdict():
    g = _g()
    ok = (_c(0.012, 0.01, 5), _c(0.003, 0.3, 4))
    assert g.verdict_s3l(*ok, _c(0.006, 0.02, 4), _c(0.006, 0.03, 5)) == "S3L-E-WIN+MAP+CONTENT"
    assert g.verdict_s3l(_c(0.004, 0.4, 3), _c(0.0, 1.0, 3), _c(0.0, 1.0, 3), _c(0.0, 1.0, 3)) == "S3L-E-NULL"
    assert g.verdict_s3l(_c(-0.02, 0.01, 1), _c(-0.02, 0.01, 1), _c(0.0, 1.0, 3), _c(0.0, 1.0, 3)) == "S3L-E-HURT"


def test_cli():
    a = _g().build_parser().parse_args(["--search", "--fold", "Cricket"])
    assert a.search and a.fold == "Cricket"
