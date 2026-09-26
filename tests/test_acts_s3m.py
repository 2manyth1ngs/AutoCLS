"""WP17-ACTS S3-M（results/acts/s3m/PROTOCOL.md）——先于实现。覆盖：降秩回归轴、留一个源解释方差、选择规则、读数码、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _g():
    return importlib.import_module("scripts.100_acts_s3m")


def _toy(seed=0, n_src=8, per=6):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.array([f"s{i}" for i in range(n_src)]), per)
    U = rng.normal(size=(len(groups), 4))
    B = np.linalg.qr(rng.normal(size=(30, 2)))[0]                                   # 两个 E 可预测方向
    R = U[:, :2] @ np.diag([3.0, 1.5]) @ B.T + 0.3 * rng.normal(size=(len(groups), 30))
    return U, R, groups, B


def test_rrr_recovers_predictable_axes():
    g = _g()
    U, R, groups, B = _toy()
    fit = g.rrr_fit(U, R, lam=0.1, k=2)
    Q = fit["Q"]
    assert Q.shape == (30, 2)
    assert np.linalg.svd(B.T @ Q, compute_uv=False).min() > 0.95                        # 子空间对齐
    ah = g.rrr_coords(fit, U)
    assert ah.shape == (len(U), 2)


def test_loso_ev_prefers_true_rank():
    g = _g()
    U, R, groups, _B = _toy(1)
    ev = {k: g.loso_ev(U, R, groups, lam=1.0, k=k) for k in (1, 2, 3)}
    assert ev[2] > ev[1] > 0 and ev[3] <= ev[2] + 0.01


def test_select():
    g = _g()
    tab = {("sh4", 10.0, 1): 0.05, ("sh4", 100.0, 1): 0.05, ("pt4", 1.0, 2): 0.04}
    assert g.select_rrr(tab) == ("sh4", 100.0, 1)                                       # 平局取小 K、大 λ
    assert g.select_rrr({("sh4", 1.0, 1): -0.01}) is None                                # EV ≤ 0 → 无可预测轴


def _c(d, p, npos):
    return {"d": d, "p": p, "npos": npos}


def test_verdict():
    g = _g()
    ok = (_c(0.012, 0.01, 5), _c(0.003, 0.3, 4))
    assert g.verdict_s3m(*ok, _c(0.006, 0.02, 4), _c(0.006, 0.03, 5)) == "S3M-E-WIN+GAIN+CONTENT"
    assert g.verdict_s3m(_c(0.004, 0.4, 3), _c(0.0, 1.0, 3), _c(0.0, 1.0, 3), _c(0.0, 1.0, 3)) == "S3M-E-NULL"


def test_cli():
    a = _g().build_parser().parse_args(["--train", "--kind", "EP", "--fold", "Cricket", "--seed", "1"])
    assert a.train and a.kind == "EP" and a.fold == "Cricket" and a.seed == 1
