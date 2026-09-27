"""WP17-ACTS S3-Q（results/acts/s3q/PROTOCOL.md）——先于实现。覆盖：标准化余弦、近邻画像、k / α 选择、精确符号翻转、读数码、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _g():
    return importlib.import_module("scripts.104_acts_s3q")


def test_std_cos():
    g = _g()
    Xtr = np.array([[1.0, 0.0], [3.0, 0.0], [2.0, 2.0], [2.0, -2.0]])
    s = g.std_cos(Xtr, np.array([3.0, 0.0]))
    assert s.shape == (4,) and int(np.argmax(s)) == 1 and abs(s[1] - 1.0) < 1e-9


def test_knn_profile():
    g = _g()
    Z = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    sim = np.array([0.1, 0.9, 0.5])
    p, nb = g.knn_profile(Z, sim, 2)
    assert list(nb) == [1, 2] and np.allclose(p, [1.0, 1.5])


def test_choose_k_and_alpha():
    g = _g()
    assert g.choose_k({3: 0.02, 5: 0.01, 10: 0.01, 20: 0.03}) == 10          # 平局取大 k
    assert g.choose_alpha_sp({1.0: 0.20, 10.0: 0.25, 100.0: 0.25}) == 100.0  # Spearman 最大，平局取大 α


def test_signflip():
    g = _g()
    assert abs(g.signflip_p(np.ones(5)) - 2 / 32) < 1e-12
    assert g.signflip_p(np.zeros(4)) == 1.0
    d = np.array([0.3, -0.1, 0.2, 0.25, -0.05, 0.4])
    p = g.signflip_p(d)
    assert 0 < p < 1


def test_spearman_rows():
    g = _g()
    P = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], float)
    T = np.array([[10, 20, 30, 40], [1, 2, 3, 4]], float)
    assert np.allclose(g.spearman_rows(P, T), [1.0, -1.0])


def _c(d, p):
    return {"d": d, "p": p}


def test_verdict():
    g = _g()
    assert g.verdict_s3q(_c(0.005, 0.01), _c(0.004, 0.02), _c(0.05, 0.01)) == "S3Q-RETR-WIN+SPACE"
    assert g.verdict_s3q(_c(0.005, 0.01), _c(0.004, 0.2), _c(0.01, 0.5)) == "S3Q-RETR-VALUE"
    assert g.verdict_s3q(_c(0.002, 0.01), _c(0.004, 0.02), _c(0.01, 0.5)) == "S3Q-RETR-CONTENT"
    assert g.verdict_s3q(_c(0.001, 0.4), _c(-0.001, 0.6), _c(0.0, 1.0)) == "S3Q-RETR-NULL"


def test_cli():
    a = _g().build_parser().parse_args(["--analyze"])
    assert a.analyze and not a.smoke
