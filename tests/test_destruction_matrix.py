"""破坏矩阵（scripts/14_destruction_matrix.py）纯逻辑测试。

期望值手工推导：破坏判定带、速率表、SS 分解（纯主效应表 vs 纯交互表）、
mixed CLS 判定、判读带边界。
"""
import importlib

dm = importlib.import_module("scripts.14_destruction_matrix")


class TestDestructionFlag:
    def test_band_semantics(self):
        # mean=1.0, band=0.25 → 阈 0.75（二进制精确）：0.74 破坏、0.75 不破坏（严格小于）
        assert dm.destruction_flag(0.74, 1.0, 0.25) == 1
        assert dm.destruction_flag(0.75, 1.0, 0.25) == 0


class TestRateTable:
    def test_mean_over_tasks(self):
        p = dm.rate_table({("a", "d1"): [1, 0], ("a", "d2"): [1, 1]})
        assert p["a"]["d1"] == 0.5 and p["a"]["d2"] == 1.0


class TestSSDecomposition:
    def test_pure_cls_effect(self):
        # 行恒定（cls 决定一切）→ TDS=0
        p = {"a": {"d1": 1.0, "d2": 1.0}, "b": {"d1": 0.0, "d2": 0.0}}
        dec = dm.ss_decomposition(p)
        assert dec["share_cls"] == 1.0 and dec["tds"] == 0.0

    def test_pure_interaction(self):
        # 完全交叉翻转（行/列均值全等）→ 全部落交互，TDS=1
        p = {"a": {"d1": 1.0, "d2": 0.0}, "b": {"d1": 0.0, "d2": 1.0}}
        dec = dm.ss_decomposition(p)
        assert dec["share_inter"] == 1.0 and dec["tds"] == 1.0

    def test_incomplete_dataset_dropped(self):
        p = {"a": {"d1": 1.0, "d2": 0.5}, "b": {"d1": 0.0}}
        dec = dm.ss_decomposition(p)
        assert dec["dropped_datasets"] == ["d2"] and dec["n_datasets"] == 1


class TestMixedCls:
    def test_mixed_requires_both_sides(self):
        row_mixed = {f"d{i}": 1.0 for i in range(3)} | {f"e{i}": 0.0 for i in range(3)}
        row_uniform = {f"d{i}": 1.0 for i in range(6)}
        p = {"m": row_mixed, "u": row_uniform}
        out = dm.mixed_cls(p)
        assert [m["cls_key"] for m in out] == ["m"]

    def test_thresholds(self):
        # 破坏 2 数据集（<3）→ 不算 mixed
        p = {"m": {"d1": 1.0, "d2": 0.6, "e1": 0.0, "e2": 0.0, "e3": 0.0}}
        assert dm.mixed_cls(p) == []


class TestVerdict:
    def test_bands(self):
        assert dm.verdict(0.30, 3) == "established"
        assert dm.verdict(0.30, 2) == "edge"        # mixed 不足但 TDS ≥0.10
        assert dm.verdict(0.05, 2) == "edge"        # mixed=2 单独触发边缘
        assert dm.verdict(0.05, 1) == "not_established"
        assert dm.verdict(None, 5) == "undecidable"

    def test_boundary_exact(self):
        assert dm.verdict(0.25, 3) == "established"
        assert dm.verdict(0.10, 0) == "edge"
