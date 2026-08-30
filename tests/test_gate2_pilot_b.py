"""闸 2 试点 B（scripts/23_gate2_pilot_b.py）纯逻辑测试。

协议 results/gate2_pilot_b/PROTOCOL.md：连续增益预测/MAE、通用平滑性、
赢家选拔、判读带。期望值手工推导。
"""
import importlib

import numpy as np
import pytest

gb = importlib.import_module("scripts.23_gate2_pilot_b")


class TestGainPredict:
    def test_weighted_average_and_mae(self):
        gains = {"A": np.array([0.1, -0.2]), "B": np.array([0.3, 0.0])}
        pred = gb.gain_predict(gains, ["A", "B"], {"A": 1.0, "B": 3.0})
        np.testing.assert_allclose(pred, [0.25, -0.05])
        assert gb.gain_mae(pred, np.array([0.25, 0.15])) == pytest.approx(0.10)


class TestSmoothnessGeneric:
    def test_planted_signal(self):
        rng = np.random.default_rng(0)
        pos = np.sort(rng.uniform(0, 10, 12))
        F = {f"T{i}": np.array([p]) for i, p in enumerate(pos)}
        prof = {f"T{i}": np.array([p, 2 * p]) for i, p in enumerate(pos)}
        rho, p = gb.smoothness_generic(
            F, prof, dist_fn=lambda a, b: float(np.abs(a - b).mean()),
            n_perm=2000, seed=1)
        assert rho > 0.5 and p < 0.01

    def test_deterministic(self):
        rng = np.random.default_rng(2)
        F = {f"T{i}": rng.normal(size=3) for i in range(10)}
        prof = {f"T{i}": rng.normal(size=5) for i in range(10)}
        fn = lambda a, b: float(np.abs(a - b).mean())  # noqa: E731
        assert gb.smoothness_generic(F, prof, dist_fn=fn, n_perm=300, seed=4) \
            == gb.smoothness_generic(F, prof, dist_fn=fn, n_perm=300, seed=4)


class TestWinner:
    def test_lowest_p_positive_direction(self):
        combos = {
            "E|T1": {"d1_p": 0.40, "direction": 1, "smooth_rho": 0.2},
            "P|T2": {"d1_p": 0.02, "direction": 1, "smooth_rho": 0.5},
            "G|T1": {"d1_p": 0.01, "direction": -1, "smooth_rho": 0.6},  # 方向负排除
        }
        assert gb.select_winner(combos) == "P|T2"

    def test_tiebreak_by_rho(self):
        combos = {
            "A|T1": {"d1_p": 0.10, "direction": 1, "smooth_rho": 0.3},
            "B|T2": {"d1_p": 0.10, "direction": 1, "smooth_rho": 0.7},
        }
        assert gb.select_winner(combos) == "B|T2"

    def test_all_negative_returns_none(self):
        combos = {"A|T1": {"d1_p": 0.01, "direction": -1, "smooth_rho": 0.9}}
        assert gb.select_winner(combos) is None


class TestVerdictB:
    def test_bands(self):
        assert gb.verdict_pilot_b(0.01, 0.5) == "strong"
        assert gb.verdict_pilot_b(0.20, 0.005) == "strong"   # 平滑性 p<0.01 单独触发
        assert gb.verdict_pilot_b(0.20, 0.05) == "weak"
        assert gb.verdict_pilot_b(0.50, 0.05) == "null"
        assert gb.verdict_pilot_b(None, None) == "null"      # 无正方向赢家

    def test_boundary(self):
        assert gb.verdict_pilot_b(0.05, 0.02) == "weak"      # 0.05 不含
        assert gb.verdict_pilot_b(0.30, 0.02) == "null"
