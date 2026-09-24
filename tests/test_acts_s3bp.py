"""WP17-ACTS S3-B′（results/acts/s3bp/PROTOCOL.md）——先于实现。覆盖：只用训练行拟合的标准化 + PCA、field 设计矩阵、读数码、命令行。"""
from __future__ import annotations

import importlib

import numpy as np


def _g():
    return importlib.import_module("scripts.87_acts_s3bp")


def test_pca_fit_only_on_train():
    g = _g()
    rng = np.random.default_rng(0)
    E = rng.normal(size=(30, 10)); tr = np.arange(20)
    f1 = g.pca_fit(E[tr], 3)
    E2 = E.copy(); E2[20:] = 1e6
    f2 = g.pca_fit(E2[tr], 3)
    assert np.allclose(g.pca_apply(f1, E[:20]), g.pca_apply(f2, E2[:20]))
    U = g.pca_apply(f1, E[tr])
    assert U.shape == (20, 3) and np.allclose(U.mean(0), 0, atol=1e-9)
    assert abs(np.corrcoef(U[:, 0], U[:, 1])[0, 1]) < 1e-6


def test_field_design():
    g = _g()
    phi = np.array([[1.0, 0.0], [0.0, 1.0]]); U = np.array([[2.0, 3.0], [5.0, 7.0]])
    X = g.field_design(phi, U)
    assert X.shape == (2, 2 + 2 * 2)
    assert np.allclose(X[0], [1, 0, 2, 0, 3, 0]) and np.allclose(X[1], [0, 1, 0, 5, 0, 7])


def test_verdict():
    g = _g()
    base = {"F1": False, "F2": False, "F3": False, "F4": False}
    assert g.verdict_fe({"pt": dict(base), "sh": dict(base)}) == "FE-INERT"
    assert g.verdict_fe({"pt": {**base, "F1": True}, "sh": {**base, "F2": True}}) == "FE-CONTENT+VALUE"
    assert g.verdict_fe({"pt": {**base, "F3": True, "F4": True}, "sh": dict(base)}) == "FE-REPLACE+SPACE"


def test_cli():
    assert _g().build_parser().parse_args(["--analyze"]).analyze is True
