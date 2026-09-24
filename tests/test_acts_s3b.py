"""WP17-ACTS S3-B（results/acts/s3b/PROTOCOL.md）——先于实现。覆盖：TS2Vec iters 规则、窗口采样确定性、均值池化 E、集内 / 跨集余弦、
cross-E 指派、验证源选择、折 TS2Vec fit 池排除留出源、Holm、门与读数码、SetTF 回归头训练（合成数据，loss 下降 / 同 seed 确定 / 子采窗口）。零真实数据。"""
from __future__ import annotations

import importlib

import numpy as np
import pytest


def _g():
    return importlib.import_module("scripts.86_acts_s3b")


class TestPure:
    def test_iters_rule(self):
        g = _g()
        assert g.ts2vec_iters(100_000) == 200 and g.ts2vec_iters(100_001) == 600

    def test_window_idx(self):
        g = _g()
        a = g.window_idx(500, "HW_a00", 100); b = g.window_idx(500, "HW_a00", 100)
        assert np.array_equal(a, b) and len(a) == 100 and np.all(np.diff(a) > 0)
        assert not np.array_equal(a, g.window_idx(500, "HW_a01", 100))
        assert np.array_equal(g.window_idx(40, "x", 100), np.arange(40))

    def test_mean_pool(self):
        g = _g()
        r = np.arange(2 * 3 * 4, dtype=np.float16).reshape(2, 3, 4)
        assert np.allclose(g.mean_pool_E(r), r.astype(np.float64).mean((0, 1)))

    def test_cosines(self):
        g = _g()
        E = np.array([[1, 0], [1, 0.01], [0, 1], [0.01, 1]], float)
        w, b = g.cosine_within_between(E, np.array(["A", "A", "B", "B"]))
        assert w > 0.99 and b < 0.05

    def test_cross_assign(self):
        g = _g()
        tasks = ["A_a00", "A_a01", "B_a00", "C_a00", "C_a01", "C_a02"]
        groups = np.array(["A", "A", "B", "C", "C", "C"])
        m = g.cross_assign(tasks, groups, ["A", "B", "C"])
        assert [tasks[j] for j in m] == ["B_a00", "B_a00", "C_a00", "A_a00", "A_a01", "A_a00"]

    def test_val_source(self):
        g = _g()
        order = ["A", "B", "C", "D"]
        assert g.val_source(order, "A") == "B" and g.val_source(order, "D") == "A"

    def test_fit_pool_excludes_heldout(self):
        g = _g()
        assert g.fit_pool_sources(["A", "B", "C"], "B") == ["A", "C"]
        assert g.fit_pool_sources(["A", "B", "C"], "ALL") == ["A", "B", "C"]

    def test_split_long(self):
        g = _g()
        x = np.full((2, 25, 1), np.nan, np.float32); x[0, :25, 0] = 1.0; x[1, :8, 0] = 2.0
        y = g.split_long(x, 10)
        assert y.shape == (4, 10, 1)                       # 行 0 → 3 段；行 1 → 1 段（其余全 NaN 丢弃）
        assert np.isnan(y[2, 5:, 0]).all() and not np.isnan(y[2, :5, 0]).any()
        assert g.split_long(x, 30) is not None and g.split_long(x, 30).shape == (2, 25, 1)

    def test_holm(self):
        g = _g()
        assert np.allclose(g.holm([0.01, 0.04, 0.03, 0.5]), [0.04, 0.09, 0.09, 0.5])


class TestVerdict:
    def _e(self, ci_lo=-0.1, p=0.5, d=0.0, ub=0.02):
        return {"rho_diff_ci_lo": ci_lo, "p_vs_land": p, "delta_vs_land": d, "vs_probe_ub": ub}

    def test_codes(self):
        g = _g()
        E = {k: self._e() for k in ("pt_mean", "sh_mean", "pt_settf", "sh_settf")}
        C = {k: {"delta_vs_probe": 0.0, "p_vs_probe": 0.9} for k in ("pt_mean+p", "sh_mean+p", "pt_settf+p", "sh_settf+p")}
        assert g.verdict_s3b(E, C) == "S3B-E-NOT-BETTER"
        E["sh_mean"] = self._e(ci_lo=0.01, p=0.001, d=0.002)
        assert g.verdict_s3b(E, C) == "S3B-E-WIN"
        E["sh_mean"] = self._e(ub=0.004)
        C["pt_mean+p"] = {"delta_vs_probe": 0.003, "p_vs_probe": 0.001}
        assert g.verdict_s3b(E, C) == "S3B-E-NOT-BETTER+REPLACE+ADD"

    def test_holm_applied_to_main_gate(self):
        g = _g()
        E = {k: self._e(ci_lo=0.01, p=0.02, d=0.002) for k in ("pt_mean", "sh_mean", "pt_settf", "sh_settf")}
        C = {k: {"delta_vs_probe": 0.0, "p_vs_probe": 0.9} for k in ("a", "b")}
        assert g.verdict_s3b(E, C) == "S3B-E-NOT-BETTER"      # 4 × 0.02 Holm → 0.08
        E["pt_mean"]["p_vs_land"] = 0.005
        assert g.verdict_s3b(E, C) == "S3B-E-WIN"             # 0.005 × 4 = 0.02


class TestSetTF:
    def _data(self, n=24, seed=0):
        rng = np.random.default_rng(seed)
        a = rng.normal(size=(n, 3))
        reprs = []
        for i in range(n):
            base = rng.normal(size=(1, 1, 8)) * 0.1
            base[0, 0, :3] += a[i]
            reprs.append((base + rng.normal(scale=0.05, size=(40, 6, 8))).astype(np.float16))
        return reprs, a

    def test_training_reduces_loss_and_deterministic(self):
        g = _g()
        reprs, a = self._data()
        tr, va = np.arange(18), np.arange(18, 24)
        kw = dict(dim_input=8, nz=16, max_epochs=30, patience=30, lr=3e-3, batch_tasks=6, sub_windows=8, device="cpu")
        r1 = g.train_settf(reprs, a, tr, va, extra=None, seed=0, **kw)
        r2 = g.train_settf(reprs, a, tr, va, extra=None, seed=0, **kw)
        assert r1["train_loss"][-1] < r1["train_loss"][0]
        assert np.allclose(r1["predict"](np.arange(24)), r2["predict"](np.arange(24)), atol=1e-5)
        assert r1["predict"](np.arange(24)).shape == (24, 3)

    def test_extra_features_widen_head(self):
        g = _g()
        reprs, a = self._data(n=12)
        extra = np.random.default_rng(1).normal(size=(12, 5))
        r = g.train_settf(reprs, a, np.arange(8), np.arange(8, 12), extra=extra, seed=0, dim_input=8, nz=16, max_epochs=2,
                          patience=5, lr=1e-3, batch_tasks=4, sub_windows=4, device="cpu")
        assert r["head_in"] == 16 + 5 and r["predict"](np.arange(12)).shape == (12, 3)

    def test_subsample_windows(self):
        g = _g()
        rng = np.random.default_rng(0)
        x = np.zeros((10, 3, 2)); assert g.subsample(x, 32, rng).shape[0] == 10
        x = np.zeros((100, 3, 2)); assert g.subsample(x, 32, rng).shape[0] == 32


def test_cli():
    ap = _g().build_parser()
    a = ap.parse_args(["--train-settf", "--arm", "sh", "--fold", "Adiac", "--combo"])
    assert a.train_settf and a.arm == "sh" and a.fold == "Adiac" and a.combo
