"""闸 2 试点（scripts/22_gate2_pilot.py）纯逻辑测试。

协议 results/gate2_pilot/PROTOCOL.md：核插值/均匀锚、画像 Brier、
平滑性置换、判读带。期望值手工推导。
"""
import importlib

import numpy as np
import pytest

gp = importlib.import_module("scripts.22_gate2_pilot")


class TestAnchorPredict:
    def _profiles(self):
        # 3 个锚任务 × 4 cls，状态码 0/1/2
        return {"A": np.array([0, 0, 1, 2]), "B": np.array([1, 1, 1, 2]),
                "C": np.array([2, 2, 2, 2])}

    def test_uniform_average(self):
        pred = gp.anchor_predict(self._profiles(), ["A", "B"],
                                 weights={"A": 1.0, "B": 1.0})
        # cls0: A=0,B=1 → [0.5,0.5,0]; cls3: 均 2 → [0,0,1]
        np.testing.assert_allclose(pred[0], [0.5, 0.5, 0.0])
        np.testing.assert_allclose(pred[3], [0.0, 0.0, 1.0])

    def test_weighted_pulls_toward_heavy_anchor(self):
        pred = gp.anchor_predict(self._profiles(), ["A", "C"],
                                 weights={"A": 3.0, "C": 1.0})
        np.testing.assert_allclose(pred[0], [0.75, 0.0, 0.25])

    def test_kernel_weights_limits(self):
        d = {"A": 1.0, "B": 2.0}
        w_inf = gp.kernel_w(d, sigma=1e9)          # σ→∞ 退化为均匀
        assert w_inf["A"] == pytest.approx(w_inf["B"], rel=1e-6)
        w = gp.kernel_w(d, sigma=1.0)
        assert w["A"] > w["B"] > 0


class TestProfileBrier:
    def test_value(self):
        pred = np.array([[1.0, 0, 0], [0, 0.5, 0.5]])
        actual = np.array([0, 2])
        # cls0 完美 0；cls1: (0)²+(0.5)²+(0.5)² = 0.5 → 均值 0.25
        assert gp.profile_brier(pred, actual) == pytest.approx(0.25)


class TestSmoothness:
    def test_planted_correlation_detected(self):
        rng = np.random.default_rng(0)
        # 任务在 1 维流形上：E 距离与画像距离同源
        pos = np.sort(rng.uniform(0, 10, 12))
        E = {f"T{i}": np.array([p, 0.0]) for i, p in enumerate(pos)}
        prof = {f"T{i}": (np.arange(8) < p) .astype(int) for i, p in enumerate(pos)}
        rho, p = gp.smoothness_test(E, prof, n_perm=2000, seed=1)
        assert rho > 0.5 and p < 0.01

    def test_null_no_signal(self):
        rng = np.random.default_rng(2)
        E = {f"T{i}": rng.normal(size=4) for i in range(12)}
        prof = {f"T{i}": rng.integers(0, 3, 8) for i in range(12)}
        rho, p = gp.smoothness_test(E, prof, n_perm=2000, seed=1)
        assert p > 0.05

    def test_deterministic(self):
        rng = np.random.default_rng(3)
        E = {f"T{i}": rng.normal(size=4) for i in range(10)}
        prof = {f"T{i}": rng.integers(0, 3, 6) for i in range(10)}
        r1 = gp.smoothness_test(E, prof, n_perm=500, seed=7)
        r2 = gp.smoothness_test(E, prof, n_perm=500, seed=7)
        assert r1 == r2


class TestVerdictPilot:
    def test_bands(self):
        assert gp.verdict_pilot(0.01, +1) == "positive"
        assert gp.verdict_pilot(0.30, +1) == "lean_positive"
        assert gp.verdict_pilot(0.30, -1) == "negative"
        assert gp.verdict_pilot(0.05, +1) == "lean_positive"   # 0.05 不含


class TestAnchorSampling:
    def test_excludes_held_and_deterministic(self):
        tasks = [f"T{i}" for i in range(16)]
        s1 = gp.sample_anchors(tasks, held="T3", k=3, seed=5)
        s2 = gp.sample_anchors(tasks, held="T3", k=3, seed=5)
        assert s1 == s2 and "T3" not in s1 and len(set(s1)) == 3
        assert gp.sample_anchors(tasks, held="T3", k=3, seed=6) != s1
