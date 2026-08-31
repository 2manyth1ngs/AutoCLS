"""锚子采样模拟（scripts/25_anchor_sim.py）纯逻辑测试。

协议 results/anchor_sim/PROTOCOL.md：向量化置换检验与 scripts/21 等价、
子格抽样确定性、三旋钮读数、闸门操作特性、工作点选择规则、真值对账硬闸。
期望值手工推导。
"""
import importlib

import numpy as np
import pytest

asim = importlib.import_module("scripts.25_anchor_sim")
tp = importlib.import_module("scripts.21_tasklevel_precheck")


# ---------------------------------------------------------------- T2 批量版等价
class TestT2Batch:
    def test_matches_scalar_reference(self):
        rng = np.random.default_rng(7)
        for _ in range(5):
            mat = rng.integers(0, 3, size=(10, 8))
            ref, _comp = tp.t2_statistic(mat)
            got = asim.t2_statistic_batch(mat[None])[0]
            assert np.isclose(got, ref, rtol=1e-12)

    def test_batch_equals_per_item(self):
        rng = np.random.default_rng(11)
        mats = rng.integers(0, 3, size=(6, 5, 3))
        batch = asim.t2_statistic_batch(mats)
        for b in range(6):
            assert np.isclose(batch[b], tp.t2_statistic(mats[b])[0], rtol=1e-12)


# ---------------------------------------------------------------- 向量化置换检验
class TestPermTestVec:
    def test_constant_matrix_p_is_one(self):
        mat = np.ones((6, 3), dtype=int)
        res = asim.perm_test_vec(mat, n_perm=500,
                                 rng=np.random.default_rng(0))
        assert res["t2_obs"] == 0.0
        assert res["p"] == 1.0

    def test_matches_reference_machine(self):
        # 中等异质矩阵：与 scripts/21 的逐次置换机在 MC 容差内同 p
        mat = np.array([[0, 0, 2], [0, 0, 2], [0, 1, 2], [1, 1, 1],
                        [0, 2, 2], [1, 0, 2], [0, 1, 1], [2, 0, 1]])
        ref = tp.perm_test(mat, n_perm=4000, seed=123)
        got = asim.perm_test_vec(mat, n_perm=4000,
                                 rng=np.random.default_rng(9))
        assert np.isclose(got["t2_obs"], ref["t2_obs"], rtol=1e-12)
        assert abs(got["p"] - ref["p"]) < 0.05

    def test_deterministic_given_rng_seed(self):
        mat = np.array([[0, 2, 1], [1, 0, 2], [2, 2, 0], [0, 1, 1]])
        p1 = asim.perm_test_vec(mat, n_perm=1000,
                                rng=np.random.default_rng(42))["p"]
        p2 = asim.perm_test_vec(mat, n_perm=1000,
                                rng=np.random.default_rng(42))["p"]
        assert p1 == p2

    def test_two_anchor_granularity_floor(self):
        # [[0,2]]*5：真 p = 2/2^5 = 0.0625（恒等+全镜像两种排列复现 T2_obs）
        mat = np.array([[0, 2]] * 5)
        res = asim.perm_test_vec(mat, n_perm=4000,
                                 rng=np.random.default_rng(3))
        assert 0.03 < res["p"] < 0.11
        assert res["p"] > asim.ALPHA_DEPLOY   # a=2 结构性地板（协议 PA2）


class TestMinAttainableP:
    def test_analytic_floor(self):
        assert asim.min_attainable_p(5) == pytest.approx(0.0625)
        assert asim.min_attainable_p(7) > asim.ALPHA_DEPLOY    # 0.0156
        assert asim.min_attainable_p(8) < asim.ALPHA_DEPLOY    # 0.0078


# ---------------------------------------------------------------- 子格抽样
class TestDrawSubpanel:
    def test_shapes_sorted_unique_in_range(self):
        rng = np.random.default_rng(5)
        t_idx, c_idx = asim.draw_subpanel(rng, n_tasks=8, n_cls=10, a=3, j=6)
        assert len(t_idx) == 3 and len(c_idx) == 6
        assert list(t_idx) == sorted(set(t_idx))
        assert list(c_idx) == sorted(set(c_idx))
        assert all(0 <= t < 8 for t in t_idx)
        assert all(0 <= c < 10 for c in c_idx)

    def test_deterministic_given_seed(self):
        d1 = asim.draw_subpanel(np.random.default_rng(77), n_tasks=8,
                                n_cls=10, a=2, j=5)
        d2 = asim.draw_subpanel(np.random.default_rng(77), n_tasks=8,
                                n_cls=10, a=2, j=5)
        assert np.array_equal(d1[0], d2[0]) and np.array_equal(d1[1], d2[1])

    def test_j_equals_full_catalog(self):
        _t, c_idx = asim.draw_subpanel(np.random.default_rng(1), n_tasks=8,
                                       n_cls=10, a=2, j=10)
        assert list(c_idx) == list(range(10))


# ---------------------------------------------------------------- 旋钮读数
class TestDestroyHat:
    def test_hand_computed_share(self):
        # mat: (n_cls=4, n_tasks=3)；destroy 状态码 = 2
        mat = np.array([[0, 1, 2],
                        [2, 2, 2],
                        [0, 0, 0],
                        [1, 2, 0]])
        sub = mat[np.ix_([1, 3], [0, 2])]   # 行 1,3 × 列 0,2 → [[2,2],[1,0]]
        assert asim.destroy_share(sub) == pytest.approx(2 / 4)

    def test_no_destroy(self):
        assert asim.destroy_share(np.zeros((2, 2), dtype=int)) == 0.0


# ---------------------------------------------------------------- 闸门操作特性
class TestGateMetrics:
    def test_hand_computed_tpr_fpr(self):
        fire = {"A": 0.9, "B": 0.7, "C": 0.1, "D": 0.05, "E": 0.3}
        m = asim.gate_metrics(fire, truth_pass={"A", "B"})
        assert m["tpr"] == pytest.approx(0.8)
        assert m["fpr"] == pytest.approx((0.1 + 0.05 + 0.3) / 3)


# ---------------------------------------------------------------- 工作点选择规则
def _pt(tpr, fpr, mae):
    return {"tpr": tpr, "fpr": fpr, "destroy_mae_pooled": mae}


class TestSelectWorkpoint:
    GOOD = _pt(0.9, 0.05, 0.05)

    def test_min_cost_wins(self):
        pts = {(2, 10): self.GOOD, (4, 5): self.GOOD, (3, 5): self.GOOD}
        assert asim.select_workpoint(pts) == (3, 5)      # 成本 15 < 20

    def test_tie_prefers_fewer_anchors(self):
        pts = {(4, 5): self.GOOD, (2, 10): self.GOOD}    # 成本同 20
        assert asim.select_workpoint(pts) == (2, 10)

    def test_each_band_excludes(self):
        pts = {(2, 5): _pt(0.79, 0.05, 0.05),    # TPR 差一线
               (3, 5): _pt(0.9, 0.16, 0.05),     # FPR 超
               (4, 5): _pt(0.9, 0.05, 0.11)}     # MAE 超
        assert asim.select_workpoint(pts) is None

    def test_boundary_inclusive(self):
        pts = {(2, 5): _pt(0.80, 0.15, 0.10)}    # 恰在带上：可行
        assert asim.select_workpoint(pts) == (2, 5)


# ---------------------------------------------------------------- 基线带精确枚举
class TestBandErrorStats:
    def test_constant_vector_zero_error(self):
        stats = asim.band_error_stats(np.full(8, 0.04), a=2)
        assert stats["n_subsets"] == 28                  # C(8,2)
        assert stats["median"] == 0.0 and stats["p90"] == 0.0

    def test_single_anchor_hand_case(self):
        vals = np.array([1.0, 1.0, 1.0, 3.0])            # 均值 1.5
        stats = asim.band_error_stats(vals, a=1)
        assert stats["n_subsets"] == 4
        assert stats["median"] == pytest.approx(1 / 3)   # |1−1.5|/1.5


# ---------------------------------------------------------------- 真值对账硬闸
class TestReconcileTruth:
    REF = {"X": {"p": 0.001, "t2_obs": 1.05},
           "Y": {"p": 0.119, "t2_obs": 0.083}}

    def test_match_passes(self):
        asim.reconcile_truth(self.REF, self.REF)         # 不抛即过

    def test_p_mismatch_raises(self):
        bad = {"X": {"p": 0.002, "t2_obs": 1.05},
               "Y": self.REF["Y"]}
        with pytest.raises(ValueError):
            asim.reconcile_truth(bad, self.REF)

    def test_t2_mismatch_raises(self):
        bad = {"X": {"p": 0.001, "t2_obs": 1.06},
               "Y": self.REF["Y"]}
        with pytest.raises(ValueError):
            asim.reconcile_truth(bad, self.REF)

    def test_missing_dataset_raises(self):
        with pytest.raises(ValueError):
            asim.reconcile_truth({"X": self.REF["X"]}, self.REF)
