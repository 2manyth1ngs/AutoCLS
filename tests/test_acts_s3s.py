"""WP17-ACTS S3-S（results/acts/s3s/PROTOCOL.md）——先于实现。覆盖：偏好相似度矩阵、对齐投影在玩具数据上能找对邻居、近邻、超参选择、读数码、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _g():
    return importlib.import_module("scripts.106_acts_s3s")


def test_pref_sim():
    g = _g()
    Z = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [4, 3, 2, 1]], float)
    S = g.pref_sim(Z)
    assert np.allclose(np.diag(S), 1) and abs(S[0, 1] - 1) < 1e-12 and abs(S[0, 2] + 1) < 1e-12


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(4), 8); h = np.tile([0, 1], 16)
    X = np.c_[h + 0.1 * rng.normal(size=32), 3 * rng.normal(size=(32, 9))]
    S = np.where(h[:, None] == h[None, :], 1.0, -1.0)
    return X, S, groups, h


def _same_rate(D, groups, h, k=3):
    hits = []
    for i in range(len(h)):
        c = np.where(groups != groups[i])[0]
        nb = c[np.argsort(D[i, c], kind="stable")[:k]]
        hits.append(np.mean(h[nb] == h[i]))
    return float(np.mean(hits))


def test_align_finds_preference_neighbors():
    g = _g()
    X, S, groups, h = _toy()
    Xs = (X - X.mean(0)) / X.std(0)
    W = g.fit_align(Xs, S, groups, r=2, lam=1e-3, steps=300, seed=0)
    U = Xs @ W
    Da = ((U[:, None] - U[None]) ** 2).sum(-1)
    Xn = Xs / np.linalg.norm(Xs, axis=1, keepdims=True)
    Dr = -(Xn @ Xn.T)
    ra, rr = _same_rate(Da, groups, h), _same_rate(Dr, groups, h)
    assert ra >= 0.8 and ra > rr + 0.1


def test_knn_dist():
    g = _g()
    Utr = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]])
    assert list(g.knn_dist(Utr, np.array([0.9, 0.1]), 2)) == [1, 0]


def test_choose_hp():
    g = _g()
    v = {(8, 1e-3): 0.02, (32, 1e-3): 0.01, (8, 1e-2): 0.01, (32, 1e-1): 0.03}
    assert g.choose_hp(v) == (8, 1e-2)                      # 平局取大 λ，再取小 r


def _c(d, p):
    return {"d": d, "p": p}


def test_verdict():
    g = _g()
    assert g.verdict_s3s(_c(0.003, 0.01), _c(0.004, 0.02)) == "S3S-ALIGN-WIN"
    assert g.verdict_s3s(_c(0.003, 0.01), _c(0.001, 0.5)) == "S3S-ALIGN-HELPS"
    assert g.verdict_s3s(_c(0.001, 0.01), _c(0.004, 0.02)) == "S3S-ALIGN-NULL"


def test_cli():
    a = _g().build_parser().parse_args(["--analyze"])
    assert a.analyze and not a.smoke
