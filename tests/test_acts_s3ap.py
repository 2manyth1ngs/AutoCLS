"""WP17-ACTS S3-A′（results/acts/s3ap/PROTOCOL.md）——先于实现。覆盖：探针特征、嵌套贪心选探针、SH / plain 选择与轨迹续跑成本、
Mk 对照、支配判定与三项读数码、50 候选共同分母 regret（random 赢家诅咒扣减）、逐任务 Spearman、富集比与源级精确置换检验。零 GPU、不读真实数据。"""
from __future__ import annotations

import importlib

import numpy as np
import pytest


def _g():
    return importlib.import_module("scripts.85_acts_s3ap")


class TestProbeFeatures:
    def test_shape_and_centering(self):
        g = _g()
        acc = np.array([[0.1, 0.2, 0.6, 0.9], [0.5, 0.5, 0.5, 0.5]])
        X = g.probe_X(acc, [0, 2, 3], np.array([0.05, 0.1]))
        assert X.shape == (2, 4)
        assert np.allclose(X[0, :3], [0.1 - 1.6 / 3, 0.6 - 1.6 / 3, 0.9 - 1.6 / 3]) and np.isclose(X[0, 3], 1.6 / 3 - 0.05)
        assert np.allclose(X[1, :3], 0.0) and np.isclose(X[1, 3], 0.4)

    def test_greedy_picks_informative_pair(self):
        g = _g()
        rng = np.random.default_rng(0)
        n, m = 120, 10
        groups = np.repeat(np.arange(12), 10)
        acc = rng.normal(0.5, 0.05, size=(n, m))
        A = np.zeros((n, 3))
        A[:, 0] = (acc[:, 3] - acc[:, 7]) * 20 + rng.normal(scale=0.05, size=n)
        order = g.greedy_order(acc, A, groups, np.arange(n), n_max=4, alpha=1.0)
        assert len(order) == 4 and len(set(order)) == 4
        assert {3, 7} <= set(order[:3])

    def test_greedy_only_uses_given_rows(self):
        g = _g()
        rng = np.random.default_rng(1)
        acc = rng.normal(size=(60, 8)); A = rng.normal(size=(60, 3)); groups = np.repeat(np.arange(6), 10)
        idx = np.arange(50)
        o1 = g.greedy_order(acc, A, groups, idx, n_max=3, alpha=10.0)
        acc2 = acc.copy(); acc2[50:] = 99.0; A2 = A.copy(); A2[50:] = -99.0
        assert o1 == g.greedy_order(acc2, A2, groups, idx, n_max=3, alpha=10.0)


class TestSelectionAndCost:
    def test_sh_select(self):
        g = _g()
        order = [0, 1, 2, 3, 4, 5, 6, 7]
        q25 = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.99, 0.0])
        q50 = np.array([0.5, 0.0, 0.9, 0.0, 0.1, 0.0, 0.0, 0.0])
        end0 = np.array([0.3, 0.0, 0.6, 0.0, 0.5, 0.0, 0.0, 0.0])
        pick, reach = g.sh_select(order, q25, q50, end0, 6, 3, 2)
        # rung0 = 0..5 @25%；按 q25 留 {0,2,4} @50%；按 q50 留 {2,0} → 终点；end0 最高 = 2
        assert pick == 2
        assert reach == {0: 1.0, 1: 0.25, 2: 1.0, 3: 0.25, 4: 0.5, 5: 0.25}

    def test_plain_select(self):
        g = _g()
        pick, reach = g.plain_select([4, 1, 2, 0], np.array([0.1, 0.9, 0.2, 0.0, 0.5]), k=3)
        assert pick == 1 and reach == {4: 1.0, 1: 1.0, 2: 1.0}

    def test_task_cost_trajectory_continuation(self):
        g = _g()
        # 探针 {1, 5} @25%；选择走到 {1: 1.0, 2: 1.0, 3: 0.5}；5 只到 0.25
        assert np.isclose(g.task_cost({1: 1.0, 2: 1.0, 3: 0.5}, [1, 5], 0.25), 1.0 + 1.0 + 0.5 + 0.25)
        assert np.isclose(g.task_cost({1: 0.25}, [1], 0.5), 0.5)

    def test_mk_orders(self):
        g = _g()
        acc = np.array([0.1, 0.9, 0.5, 0.3, 0.8])
        mu = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        a = g.mk_order_a(acc, [1, 3, 4], mu)
        b = g.mk_order_b(acc, [1, 3, 4], mu)
        assert a[:3] == [1, 0, 2]            # 探针最好 1 条 + μ 前 2（去重）
        assert b[:3] == [1, 4, 3]            # 探针按 acc 前 3


class TestDominance:
    def test_dominated(self):
        g = _g()
        base = [(3.0, 0.021), (3.75, 0.016), (5.0, 0.004)]
        assert g.dominated((4.0, 0.010), base) is False
        assert g.dominated((6.0, 0.005), base) is True
        assert g.dominated((5.0, 0.004), base) is True        # 相等也算被支配（基线成本 ≤ 且 regret ≤）
        assert g.dominated((2.0, 0.030), base) is False

    def test_verdict_probe(self):
        g = _g()
        rows = [{"label": "k2", "k": 2, "cost_rank": 0.5, "rho": 0.4, "undominated": True, "model_p": 0.01, "model_delta": 0.01},
                {"label": "k4", "k": 4, "cost_rank": 1.0, "rho": 0.72, "undominated": True, "model_p": 0.2, "model_delta": 0.003},
                {"label": "k6", "k": 6, "cost_rank": 1.5, "rho": 0.8, "undominated": True, "model_p": 0.01, "model_delta": 0.01}]
        assert g.verdict_probe(rows) == "PROBE-K4"
        rows[1]["model_p"] = 0.01
        assert g.verdict_probe(rows) == "PROBE-K4+MODEL"
        for r in rows:
            r["undominated"] = False
        assert g.verdict_probe(rows) == "PROBE-DOMINATED"

    def test_verdict_space(self):
        g = _g()
        assert g.verdict_space(0.6, 0.3, 0.004) == "SPACE-OK"
        assert g.verdict_space(0.6, 0.3, 0.02) == "SPACE-RANK-ONLY"
        assert g.verdict_space(0.3, 0.3, 0.0) == "SPACE-MU-FAIL"
        assert g.verdict_space(0.3, 0.6, 0.0) == "SPACE-B-FAIL"

    def test_verdict_headroom(self):
        g = _g()
        assert g.verdict_headroom(2.2, 0.01) == "HEADROOM-NEG"
        assert g.verdict_headroom(2.2, 0.2) == "HEADROOM-NEG-WEAK"
        assert g.verdict_headroom(1.5, 0.01) == "HEADROOM-UNIFORM"


class TestSpace:
    def test_o_adj_and_regret50(self):
        g = _g()
        end0 = np.array([0.5, 0.6, 0.4]); end1 = np.array([0.52, 0.58, 0.41])
        R0 = np.array([0.70, 0.30]); sigma = 0.02
        k20 = g.expected_max(2)
        o = g.o_adj(end0, end1, R0, sigma)
        assert np.isclose(o, max(0.58, 0.70 - k20 * sigma))
        # 30 候选：前 3 = 全部 shared，按 end0 选列 1 → 0.58
        r30, r50 = g.regret_30_50(np.array([3.0, 2.0, 1.0]), np.array([5.0, 0.0]), end0, end1, R0, sigma, k=3)
        assert np.isclose(r30, o - 0.58)
        # 50 候选：前 3 = random0, shared0, shared1；seed0 值 0.70 最高 → 选 random0，value = 0.70 − κ₃σ
        assert np.isclose(r50, o - (0.70 - g.expected_max(3) * sigma))
        assert np.isclose(g.expected_max(3), 0.8463, atol=1e-3)

    def test_task_spearman_min_n(self):
        g = _g()
        assert np.isnan(g.task_spearman(np.arange(5.0), np.arange(5.0), min_n=10))
        assert np.isclose(g.task_spearman(np.arange(12.0), np.arange(12.0) ** 2, min_n=10), 1.0)
        v = np.arange(12.0)
        assert np.isclose(g.task_spearman(v, v, min_n=10, top_half=True), 1.0)


class TestHeadroom:
    def test_enrichment(self):
        g = _g()
        E = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0], bool)
        neg = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], bool)
        assert np.isclose(g.enrichment(E, neg), (2 / 3) / (3 / 10))

    def test_exact_permutation(self):
        g = _g()
        frac = {f"S{i}": (1.0 if i < 3 else 0.0) for i in range(10)}
        p = g.perm_test_sources(frac, ["S0", "S1", "S2"])
        assert np.isclose(p, 1 / 120)          # C(10,3) = 120，只有观测分组最极端
        p2 = g.perm_test_sources({f"S{i}": 0.5 for i in range(10)}, ["S0", "S1", "S2"])
        assert np.isclose(p2, 1.0)


def test_cli():
    assert _g().build_parser().parse_args(["--analyze"]).analyze is True
