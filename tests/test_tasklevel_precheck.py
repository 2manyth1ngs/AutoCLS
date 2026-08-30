"""任务级存在性预检（scripts/21_tasklevel_precheck.py）纯逻辑测试。

协议 results/tasklevel_precheck/PROTOCOL.md：T2 统计量、行内置换零假设、
置换检验方向、判读带。期望值手工推导。
"""
import importlib

import numpy as np
import pytest

tp = importlib.import_module("scripts.21_tasklevel_precheck")


class TestT2Statistic:
    def test_hand_computed(self):
        # 2 CLS × 2 任务；状态码 0=improve 1=inert 2=destroy
        mat = np.array([[2, 1],
                        [2, 1]])
        # 任务0: destroy 率 1.0；任务1: 0.0；全局 0.5 → destroy 分量 2×0.25=0.5
        # inert 对称同 0.5；improve 全 0 → 0。T2 = 1.0
        t2, comp = tp.t2_statistic(mat)
        assert t2 == pytest.approx(1.0)
        assert comp["destroy"] == pytest.approx(0.5)
        assert comp["improve"] == pytest.approx(0.0)

    def test_uniform_zero(self):
        mat = np.array([[1, 1, 1], [2, 2, 2]])   # 每任务分布相同
        t2, _ = tp.t2_statistic(mat)
        assert t2 == pytest.approx(0.0)


class TestPermutation:
    def test_rows_preserved(self):
        mat = np.array([[0, 1, 2, 2], [1, 1, 0, 2]])
        rng = np.random.default_rng(0)
        pm = tp.permute_within_cls(mat, rng)
        for i in range(mat.shape[0]):
            assert sorted(pm[i]) == sorted(mat[i])   # 行内多重集不变

    def test_structured_small_p(self):
        # 强任务结构：任务0 全 destroy、其余全 inert，8 条 CLS 行一致
        mat = np.hstack([np.full((8, 1), 2), np.full((8, 5), 1)])
        res = tp.perm_test(mat, n_perm=2000, seed=1)
        assert res["p"] < 0.01

    def test_unstructured_large_p(self):
        rng = np.random.default_rng(3)
        mat = rng.integers(0, 3, size=(10, 12))     # iid 无任务结构
        res = tp.perm_test(mat, n_perm=2000, seed=1)
        assert res["p"] > 0.05

    def test_deterministic(self):
        mat = np.random.default_rng(5).integers(0, 3, size=(10, 8))
        r1 = tp.perm_test(mat, n_perm=500, seed=9)
        r2 = tp.perm_test(mat, n_perm=500, seed=9)
        assert r1["p"] == r2["p"] and r1["t2_obs"] == r2["t2_obs"]


class TestVerdictF:
    def test_bands(self):
        assert tp.verdict_tasklevel({"HW": 0.01, "UM": 0.5}) == "positive"
        assert tp.verdict_tasklevel({"HW": 0.5, "UM": 0.024}) == "positive"
        assert tp.verdict_tasklevel({"HW": 0.03, "UM": 0.5}) == "negative"
        assert tp.verdict_tasklevel({"HW": 0.025, "UM": 0.025}) == "negative"  # 边界不含
