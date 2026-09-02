"""E 级退化诊断的纯函数单测（实验 1；results/e_degradation_diag/PROTOCOL.md §1–§2）。"""
import numpy as np
import pytest

from autocls.comparator.e_diagnostics import (
    constancy_ratio,
    cosine_matrix,
    effective_rank,
    group_stats,
    layer_report,
    relative_dispersion,
    zscore_across,
)


class TestBasics:
    def test_cosine_matrix(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        C = cosine_matrix(X)
        assert C.shape == (3, 3)
        assert np.allclose(np.diag(C), 1.0)
        assert C[0, 1] == pytest.approx(0.0)
        assert C[0, 2] == pytest.approx(1 / np.sqrt(2))

    def test_effective_rank(self):
        rng = np.random.default_rng(0)
        iso = rng.normal(size=(2000, 5))
        assert 4.5 < effective_rank(iso) < 5.0
        rank1 = np.outer(rng.normal(size=50), rng.normal(size=5))
        assert effective_rank(rank1) == pytest.approx(1.0, abs=1e-6)
        const = np.ones((10, 5))
        assert effective_rank(const) == 0.0  # 零协方差 → 定义为 0

    def test_relative_dispersion(self):
        X = np.array([[1.0, 0.0], [1.0, 0.0]])
        assert relative_dispersion(X) == 0.0
        X2 = np.array([[1.0, 0.0], [-1.0, 0.0]])
        assert relative_dispersion(X2) == np.inf  # 均值为零向量

    def test_zscore_across(self):
        rng = np.random.default_rng(1)
        X = rng.normal(3, 2, size=(30, 4))
        Z = zscore_across(X)
        assert np.allclose(Z.mean(axis=0), 0, atol=1e-9)
        assert np.allclose(Z.std(axis=0), 1, atol=1e-9)
        Zc = zscore_across(np.ones((5, 3)))
        assert np.all(np.isfinite(Zc))

    def test_constancy_ratio(self):
        C = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        assert constancy_ratio(C, c0=np.zeros(2)) == 0.0
        C2 = np.array([[1.0, 0.0], [-1.0, 0.0]])
        assert constancy_ratio(C2, c0=np.array([0.0, 1.0])) == pytest.approx(1.0)


class TestGroupStats:
    def test_two_clusters(self):
        rng = np.random.default_rng(2)
        a = np.array([1.0, 0.0, 0.0]) + 0.01 * rng.normal(size=(5, 3))
        b = np.array([0.0, 1.0, 0.0]) + 0.01 * rng.normal(size=(5, 3))
        X = np.vstack([a, b])
        groups = ["A"] * 5 + ["B"] * 5
        s = group_stats(X, groups)
        assert s["within_cos"] > 0.99 and s["cross_cos"] < 0.05
        assert s["separation_ratio"] > 10
        assert s["min_cos"] < 0.05 and s["n_within_pairs"] == 20 and s["n_cross_pairs"] == 25

    def test_layer_report_keys(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(8, 4))
        r = layer_report(X, ["A"] * 4 + ["B"] * 4)
        for k in ("within_cos", "cross_cos", "separation_ratio", "min_cos",
                  "effective_rank", "relative_dispersion", "zscored"):
            assert k in r
        assert "separation_ratio" in r["zscored"]
