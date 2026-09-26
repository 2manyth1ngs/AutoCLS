"""WP17-ACTS S3-I（results/acts/s3i/PROTOCOL.md）——先于实现。覆盖：轴瓶颈比较器（反对称、â = 0 退化为任务盲）、E → 坐标映射只用训练侧、bootstrap、读数码、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _g():
    return importlib.import_module("scripts.96_acts_s3i")


def test_axis_comparator():
    import torch
    g = _g()
    torch.manual_seed(0); m = g.AxisComparator(20, k=3)
    a, b, t = torch.rand(6, 20), torch.rand(6, 20), torch.randn(6, 3)
    with torch.no_grad():
        p, q = m(a, b, t), m(b, a, t)
        assert torch.allclose(p + q, torch.ones(6), atol=1e-6)                          # 反对称
        h = m.heads(a) - m.heads(b)
        assert torch.allclose(m(a, b, torch.zeros(6, 3)), torch.sigmoid(h[:, 0]), atol=1e-6)   # â = 0 → 只剩 s₀
        assert torch.allclose(torch.logit(p, eps=1e-7), h[:, 0] + (h[:, 1:] * t).sum(1), atol=1e-4)


def test_axis_map_uses_train_only():
    g = _g()
    rng = np.random.default_rng(0)
    U = rng.normal(size=(40, 4)); Wt = rng.normal(size=(4, 3)); A = U @ Wt + 0.01 * rng.normal(size=(40, 3))
    groups = np.repeat(np.array(list("abcdefgh")), 5); tr = groups != "h"
    fit = g.fit_axis_map(U, A, groups, tr, alphas=(0.1, 1.0))
    A2 = A.copy(); A2[~tr] += 100.0                                                    # 改留出源的坐标不应影响映射
    fit2 = g.fit_axis_map(U, A2, groups, tr, alphas=(0.1, 1.0))
    assert np.allclose(g.predict_axis(fit, U), g.predict_axis(fit2, U))
    assert np.corrcoef(g.predict_axis(fit, U)[~tr, 0], A[~tr, 0])[0, 1] > 0.95
    fb = g.fit_axis_map(U, A, groups, tr, alphas=(0.1, 1.0), boot_seed=3)              # bootstrap 版：α 同、W 不同
    assert fb["alpha"] == fit["alpha"] and not np.allclose(g.predict_axis(fb, U), g.predict_axis(fit, U))


def _c(d, p, npos):
    return {"d": d, "p": p, "npos": npos}


def test_verdict():
    g = _g()
    ok, no = (_c(0.012, 0.01, 5), _c(0.003, 0.3, 4)), (_c(0.004, 0.4, 3), _c(0.0, 1.0, 3))
    assert g.verdict_s3i(*ok, *ok, _c(0.006, 0.02, 4)) == "S3I-ARCH-OK+E-WIN+CONTENT"
    assert g.verdict_s3i(*ok, *no, _c(0.0, 1.0, 3)) == "S3I-ARCH-OK+E-NULL"
    assert g.verdict_s3i(*no, _c(-0.02, 0.01, 1), _c(-0.02, 0.01, 1), _c(0.0, 1.0, 3)) == "S3I-ARCH-NULL+E-HURT"
    assert g.verdict_s3i(_c(0.012, 0.01, 5), _c(-0.001, 0.3, 2), *no, _c(0.0, 1.0, 3)) == "S3I-ARCH-NULL+E-NULL"   # 3 选 1 反向


def test_cli():
    a = _g().build_parser().parse_args(["--train", "--fold", "Cricket", "--seed", "2"])
    assert a.train and a.fold == "Cricket" and a.seed == 2
