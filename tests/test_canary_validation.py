"""金丝雀有效性验证（scripts/20_canary_validation.py）纯逻辑测试。

协议 results/canary_validation/PROTOCOL.md：毒哨排名/指派、线性校准、
balanced accuracy、判读带。期望值手工推导。
"""
import importlib

import numpy as np
import pytest

cv = importlib.import_module("scripts.20_canary_validation")


class TestPerClsDestroyRates:
    def test_rates(self):
        cells = [
            {"cls_key": "a", "label": "destroy"},
            {"cls_key": "a", "label": "inert"},
            {"cls_key": "b", "label": "improve"},
            {"cls_key": "b", "label": "improve"},
        ]
        r = cv.per_cls_destroy_rates(cells)
        assert r["a"] == 0.5 and r["b"] == 0.0


class TestSentinelCells:
    def test_rank_and_round_robin(self):
        rates = {"a": 0.9, "b": 0.5, "c": 0.7, "d": 0.1}
        cells = cv.sentinel_cells(rates, ["T_00", "T_01"], j=3)
        # 降序 a, c, b；轮转 _00/_01/_00
        assert cells == [("T_00", "a"), ("T_01", "c"), ("T_00", "b")]

    def test_tie_deterministic(self):
        rates = {"b": 0.5, "a": 0.5}
        c1 = cv.sentinel_cells(rates, ["T_00", "T_01"], j=2)
        c2 = cv.sentinel_cells(rates, ["T_00", "T_01"], j=2)
        assert c1 == c2                      # 并列按 key 名字定序


class TestLinearCalibrate:
    def test_perfect_line_and_clip(self):
        a, b = cv.linear_calibrate([0.0, 0.5, 1.0], [0.1, 0.3, 0.5])
        assert a == pytest.approx(0.1) and b == pytest.approx(0.4)
        assert cv.calibrated_predict(0.5, a, b) == pytest.approx(0.3)
        assert cv.calibrated_predict(5.0, a, b) == 1.0     # 截断
        assert cv.calibrated_predict(-5.0, a, b) == 0.0

    def test_degenerate_constant_s(self):
        # 训练侧 s 全同 → b=0，退化为均值先验
        a, b = cv.linear_calibrate([0.4, 0.4, 0.4], [0.1, 0.2, 0.6])
        assert b == 0.0 and a == pytest.approx(0.3)


class TestBalancedAccuracy:
    def test_value(self):
        y = np.array([1, 1, 0, 0, 0, 0])
        p = np.array([1, 0, 0, 0, 0, 1])
        # TPR=1/2, TNR=3/4 → 0.625
        assert cv.balanced_accuracy(y, p) == pytest.approx(0.625)

    def test_perfect(self):
        y = np.array([1, 0, 1])
        assert cv.balanced_accuracy(y, y) == 1.0


class TestVerdictE:
    def test_bands(self):
        assert cv.verdict_canary(True, True) == "valid"
        assert cv.verdict_canary(True, False) == "edge"
        assert cv.verdict_canary(False, True) == "edge"
        assert cv.verdict_canary(False, False) == "invalid"


class TestGates:
    def test_v1_gate(self):
        assert cv.v1_pass(0.60, 0.01) is True
        assert cv.v1_pass(0.49, 0.01) is False     # ρ<0.5
        assert cv.v1_pass(0.80, 0.06) is False     # p≥0.05
        assert cv.v1_pass(0.50, 0.049) is True     # 边界含 0.5

    def test_v2_gate(self):
        assert cv.v2_pass(0.75) is True and cv.v2_pass(0.749) is False
