"""WP17-ACTS S3-P（results/acts/s3p/PROTOCOL.md）——先于实现。覆盖：候选池装配（最终种群 ∪ 分块前 20，去重保序）、LCB 重排取前 3、读数码、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _g():
    return importlib.import_module("scripts.103_acts_s3p")


def test_pool():
    g = _g()
    final = ["a", "b", "c"]; chunk = ["c", "d", "e", "a", "f"]
    assert g.build_pool(final, chunk, n_chunk=3) == ["a", "b", "c", "d", "e"]


def test_rerank():
    g = _g()
    W = np.array([[3, 1, 2, 0], [3, 0, 2, 1], [2, 1, 3, 0]], float)                     # 3 成员 × 4 候选
    assert g.rerank(W, ["w", "x", "y", "z"], lam=1.0, k=3) == ["w", "y", "x"]


def _c(d, p, npos):
    return {"d": d, "p": p, "npos": npos}


def test_verdict():
    g = _g()
    assert g.verdict_s3p(_c(0.011, 0.02, 4), _c(0.002, 0.4, 4), _c(0.006, 0.03, 5)) == "S3P-E-WIN+CONTENT"
    assert g.verdict_s3p(_c(0.004, 0.4, 3), _c(0.0, 1.0, 3), _c(0.0, 1.0, 3)) == "S3P-E-NULL"
    assert g.verdict_s3p(_c(-0.02, 0.01, 1), _c(-0.02, 0.01, 1), _c(0.0, 1.0, 3)) == "S3P-E-HURT"


def test_cli():
    a = _g().build_parser().parse_args(["--search", "--fold", "Cricket"])
    assert a.search and a.fold == "Cricket"
