"""WP17-ACTS S3-J（results/acts/s3j/PROTOCOL.md）——先于实现。覆盖：两轮扩充标签按配置去重合并、读数码、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _g():
    return importlib.import_module("scripts.97_acts_s3j")


def test_union_aug():
    g = _g()
    a = {3: [("k1", np.ones(2), 0.5), ("k2", np.zeros(2), 0.6)]}
    b = {3: [("k2", np.full(2, 9.0), 0.9), ("k3", np.full(2, 2.0), 0.7)], 5: [("k4", np.ones(2), 0.1)]}
    u = g.union_aug(a, b)
    assert sorted(u) == [3, 5]
    vec, r = u[3]
    assert vec.shape == (3, 2) and np.allclose(r, [0.5, 0.6, 0.7])                  # 同配置只计一次，先到者为准
    assert np.allclose(vec[1], 0.0) and np.allclose(u[5][1], [0.1])


def _c(d, p, npos):
    return {"d": d, "p": p, "npos": npos}


def test_verdict():
    g = _g()
    ok, no = (_c(0.012, 0.01, 5), _c(0.003, 0.3, 4)), (_c(0.004, 0.4, 3), _c(0.0, 1.0, 3))
    assert g.verdict_s3j(*ok, *ok, _c(0.006, 0.02, 4), _c(0.006, 0.03, 5)) == "S3J-ARCH-OK+E-WIN+L3+CONTENT"
    assert g.verdict_s3j(*no, *no, _c(0.006, 0.02, 4), _c(0.0, 1.0, 3)) == "S3J-ARCH-NULL+E-NULL+L3"
    assert g.verdict_s3j(*ok, _c(-0.02, 0.01, 1), _c(-0.02, 0.01, 1), _c(0.0, 1.0, 3), _c(0.0, 1.0, 3)) == "S3J-ARCH-OK+E-HURT"


def test_cli():
    a = _g().build_parser().parse_args(["--aug-search", "--fold", "Cricket"])
    assert a.aug_search and a.fold == "Cricket"
