"""MX-1c 先验收缩（scripts/18_prior_shrinkage.py）纯逻辑测试。

协议 results/mx1c_prior_shrinkage/PROTOCOL.md：秩校准映射、Dirichlet 收缩、
抽样确定性、判读带。期望值手工推导。
"""
import importlib

import numpy as np
import pytest

ps = importlib.import_module("scripts.18_prior_shrinkage")


class TestQuantileMap:
    def test_interpolation_and_clamp(self):
        q_train = [0.1, 0.5, 0.9]
        o_train = [0.0, 0.2, 0.8]          # 已按 q 对齐
        assert ps.quantile_map(0.5, q_train, o_train) == pytest.approx(0.2)
        assert ps.quantile_map(0.3, q_train, o_train) == pytest.approx(0.1)  # 线性
        assert ps.quantile_map(-1.0, q_train, o_train) == 0.0   # 端点截断
        assert ps.quantile_map(2.0, q_train, o_train) == 0.8

    def test_unsorted_input_ok(self):
        # 映射表按 q 排序后插值：乱序输入结果不变
        a = ps.quantile_map(0.4, [0.9, 0.1, 0.5], [0.8, 0.0, 0.2])
        b = ps.quantile_map(0.4, [0.1, 0.5, 0.9], [0.0, 0.2, 0.8])
        assert a == pytest.approx(b)


class TestShrink:
    def test_formula(self):
        # (j·p̄ + m·π)/(j+m)：j=3, p̄=1/3, m=4, π=0.25 → (1+1)/7
        assert ps.shrink(1 / 3, 0.25, j=3, m=4) == pytest.approx(2 / 7)
        assert ps.shrink(0.5, 0.5, j=2, m=8) == pytest.approx(0.5)

    def test_m_zero_is_raw(self):
        assert ps.shrink(0.7, 0.0, j=5, m=0) == pytest.approx(0.7)


class TestDraws:
    def test_deterministic_and_unbiased(self):
        flags = np.array([1] * 5 + [0] * 15)     # 真破坏率 0.25
        e1 = ps.simulate_fold(flags, prior_g=0.25, prior_f=0.25,
                              j=3, m=4, n_draws=200, seed=7)
        e2 = ps.simulate_fold(flags, prior_g=0.25, prior_f=0.25,
                              j=3, m=4, n_draws=200, seed=7)
        assert e1 == e2                           # 逐位确定
        # 先验=真值时收缩应显著好于 raw
        assert e1["shrink_g"] < e1["raw"]
        assert e1["shrink_f"] == pytest.approx(e1["shrink_g"])

    def test_good_prior_beats_bad_prior(self):
        flags = np.array([1] * 10 + [0] * 10)    # 真值 0.5
        e = ps.simulate_fold(flags, prior_g=0.1, prior_f=0.5,
                             j=3, m=4, n_draws=500, seed=0)
        assert e["shrink_f"] < e["shrink_g"]      # 准先验必须赢


class TestVerdictC:
    def test_bands(self):
        assert ps.verdict_mx1c(0.01, [1, 1, 1, 1]) == "established"
        assert ps.verdict_mx1c(0.10, [1, 1, 1, -1]) == "edge"       # p<0.15 + 3 档同向
        assert ps.verdict_mx1c(0.10, [1, 1, -1, -1]) == "not_established"  # 同向不足
        assert ps.verdict_mx1c(0.50, [1, 1, 1, 1]) == "not_established"

    def test_boundary(self):
        assert ps.verdict_mx1c(0.05, [1, 1, 1, 1]) == "edge"
        assert ps.verdict_mx1c(0.15, [1, 1, 1, 1]) == "not_established"
