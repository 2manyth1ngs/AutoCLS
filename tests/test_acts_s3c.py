"""WP17-ACTS S3-C（results/acts/s3c/PROTOCOL.md）——先于实现。覆盖：best-of-3 标签、AutoCTS++ linear 配对（平局记 0、跳自配对）、
验证全有序对、Round-Robin 排序（并列按 key）、分块初始选择、搜索终选（seed 0 选 / seed 1 评）、源同号计数、两组读数码、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _g():
    return importlib.import_module("scripts.89_acts_s3c")


class TestLabels:
    def test_best_of3(self):
        g = _g()
        assert g.best_of3([0.5, None, 0.7]) == 0.7
        assert np.isnan(g.best_of3([None, None, None]))
        assert g.best_of3([0.9, 0.2, 0.3]) == 0.9

    def test_linear_pairs(self):
        g = _g()
        r = np.array([0.5, 0.5, 0.7, 0.1])
        P1 = g.linear_pairs(r, np.random.default_rng(0))
        P2 = g.linear_pairs(r, np.random.default_rng(0))
        assert P1 == P2 and len(P1) <= 4
        for a, b, y in P1:
            assert a != b
            assert y == int(r[a] > r[b])                   # 平局 (0.5, 0.5) → 0
        # 平局一定记 0：构造只有两条且相等
        assert all(y == 0 for _a, _b, y in g.linear_pairs(np.array([0.3, 0.3]), np.random.default_rng(1)))

    def test_all_ordered_pairs(self):
        g = _g()
        P = g.all_ordered_pairs(np.array([0.1, 0.2, 0.2]))
        assert len(P) == 6 and (1, 2, 0) in P and (2, 1, 0) in P and (2, 0, 1) in P


class TestRanking:
    def test_rr_order_ties_by_key(self):
        g = _g()
        P = np.array([[np.nan, 0.5, 0.9], [0.5, np.nan, 0.9], [0.1, 0.1, np.nan]])
        assert g.rr_order(P, ["b", "a", "c"]) == [1, 0, 2]      # 0 与 1 胜场相同 → key 升序

    def test_chunk_select_finds_global_top(self):
        g = _g()
        rng = np.random.default_rng(0)
        s = rng.normal(size=300)
        keys = [f"k{i:03d}" for i in range(300)]

        def pmat(idx):
            v = s[np.asarray(idx)]
            return 1.0 / (1.0 + np.exp(-(v[:, None] - v[None, :]) * 5))

        top = g.chunk_select(300, pmat, keys, chunk=50, per_chunk=10, final=10)
        assert set(top) == set(np.argsort(-s)[:10].tolist())

    def test_pick_final(self):
        g = _g()
        v0 = {"a": 0.5, "b": 0.7, "c": 0.6}
        v1 = {"a": 0.55, "b": 0.61, "c": 0.9}
        assert g.pick_final(["a", "b", "c"], v0, v1) == ("b", 0.61)


class TestVerdicts:
    def test_rank_codes(self):
        g = _g()
        assert g.verdict_rank({"C1": True, "C2": True, "C3": True, "C5": False}) == "S3-E-WIN+ID-OK"
        assert g.verdict_rank({"C1": False, "C2": True, "C3": False, "C5": False}) == "S3-LOWRANK-SUFFICES+IDENTITY"
        assert g.verdict_rank({"C1": False, "C2": False, "C3": True, "C5": True}) == "S3-E-INERT+IDENTITY+SPACE"

    def test_search_codes(self):
        g = _g()
        assert g.verdict_search({"S1": True, "S2": True, "S3": True}) == "SEARCH-WIN+CATALOG-OK"
        assert g.verdict_search({"S1": True, "S2": False, "S3": False}) == "SEARCH-BLIND+CATALOG-LOSS"
        assert g.verdict_search({"S1": False, "S2": True, "S3": True}) == "SEARCH-DEAD+CATALOG-OK"

    def test_sign_consistency(self):
        g = _g()
        d = np.array([1, 1, -1, 2, 2, 3, 0.5, 0.1])
        groups = np.array(["A", "A", "B", "B", "C", "C", "D", "D"])
        assert g.n_sources_positive(d, groups) == 4        # B 的均值 +0.5 > 0


def test_cli():
    a = _g().build_parser().parse_args(["--train", "--arm", "A", "--fold", "Cricket"])
    assert a.train and a.arm == "A" and a.fold == "Cricket"
