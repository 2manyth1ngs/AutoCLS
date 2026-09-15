"""SX 检验（每集加切片）——先于实现。覆盖：排除型类别子集策略、扩展切片构造（原切片逐字段不变、id 约定、类别子集避开原集、
组合耗尽后允许重复）、S ∈ {6, 9, 12} 嵌套行选择与考侧固定、网格计划（20 + GGS、零撞键）、φ 行（基线带来自扩展缓存）、读数码。零 GPU、不读真实缓存。"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from autocls.data.slice_strategies import ExcludingClassSubsetSlicer, RandomClassSubsetSlicer
from autocls.data.source_pool import ROLE_SOURCE, DatasetEntry, SourcePoolRegistry, slice_extension_tasks, slice_source_tasks
from autocls.data.task_slicer import slice_pool_tasks
from autocls.labeling import ORIGIN_RANDOM, ORIGIN_SHARED
from scripts import pq_common as pq

m33 = importlib.import_module("scripts.33_pq_comparator_loo")


def _g74():
    return importlib.import_module("scripts.74_tclsc_slices")


def _y(n_classes: int, per: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(np.repeat(np.arange(n_classes), per))


def _registry(name="Dset", n_classes=4, rng=(2, 4), planned=6, C=6):
    e = DatasetEntry(name=name, role=ROLE_SOURCE, domain="X", n_channels=C, n_classes=n_classes, l_enc=100,
                     n_classes_range=rng, planned_tasks=planned)
    return e, SourcePoolRegistry(entries=(e,), slice_seed=0, n_samples=64, min_samples=8)


class TestExcludingSlicer:
    def test_excludes_then_repeats_after_exhaustion(self):
        y = _y(4)                                  # 4 类 [2,4] → 11 个组合
        pool = np.arange(len(y))
        base = RandomClassSubsetSlicer(n_classes_range=(2, 4), allow_repeat=False)
        orig = base.propose(y, pool, 6, np.random.default_rng(0))
        excl = tuple(s.classes for s in orig)
        ext = ExcludingClassSubsetSlicer(n_classes_range=(2, 4), exclude=excl).propose(y, pool, 6, np.random.default_rng(1))
        assert len(ext) == 6 and all(s.tag == "ext" for s in ext)
        new_sets = [frozenset(s.classes) for s in ext]
        # 前 5 个必须避开原 6 个且互不重复；第 6 个（组合耗尽）允许重复
        assert all(fs not in {frozenset(c) for c in excl} for fs in new_sets[:5])
        assert len(set(new_sets[:5])) == 5
        # 确定性
        ext2 = ExcludingClassSubsetSlicer(n_classes_range=(2, 4), exclude=excl).propose(y, pool, 6, np.random.default_rng(1))
        assert [s.classes for s in ext2] == [s.classes for s in ext]

    def test_plenty_of_combos_never_repeats(self):
        y = _y(10)
        pool = np.arange(len(y))
        excl = ((0, 1, 2), (3, 4, 5, 6))
        ext = ExcludingClassSubsetSlicer(n_classes_range=(3, 4), exclude=excl).propose(y, pool, 12, np.random.default_rng(0))
        sets = [frozenset(s.classes) for s in ext]
        assert len(set(sets)) == 12 and not ({frozenset(c) for c in excl} & set(sets))


class TestExtensionTasks:
    def test_id_prefix_and_default_unchanged(self):
        y = _y(5)
        a = slice_pool_tasks(y, np.arange(len(y)), pool_name="D", n_tasks=3, strategy=RandomClassSubsetSlicer((2, 3)),
                             n_channels_range=(2, 3), n_total_channels=3, n_samples=32, min_samples=8, seed=0)
        b = slice_pool_tasks(y, np.arange(len(y)), pool_name="D", n_tasks=3, strategy=RandomClassSubsetSlicer((2, 3)),
                             n_channels_range=(2, 3), n_total_channels=3, n_samples=32, min_samples=8, seed=0, id_prefix="x")
        assert [t.task_id for t in a] == ["D_00", "D_01", "D_02"]
        assert [t.task_id for t in b] == ["D_x00", "D_x01", "D_x02"]
        assert [(t.classes, t.channels, t.sample_indices) for t in a] == [(t.classes, t.channels, t.sample_indices) for t in b]

    def test_extension_keeps_originals_and_avoids_their_classes(self):
        e, reg = _registry(n_classes=8, rng=(3, 5), planned=6)
        y = _y(8)
        _eff, before = slice_source_tasks(e, reg, y)
        ext = slice_extension_tasks(e, reg, y, n_new=6)
        _eff, after = slice_source_tasks(e, reg, y)
        assert before == after                                                    # 原切片逐字段不变
        assert [t.task_id for t in ext] == [f"Dset_x{i:02d}" for i in range(6)]
        assert all(pq.dataset_of(t.task_id) == "Dset" for t in ext)
        orig_sets = {frozenset(t.classes) for t in before}
        assert all(frozenset(t.classes) not in orig_sets for t in ext)             # 组合充裕 → 全部避开
        assert len({frozenset(t.classes) for t in ext}) == 6
        assert all(t.tag == "ext" for t in ext)
        for t in ext:                                                             # 样本 ⊆ 所选类别
            assert set(y[list(t.sample_indices)]) <= set(t.classes)
        assert slice_extension_tasks(e, reg, y, n_new=6) == ext                    # 确定性
        assert slice_extension_tasks(e, reg, y, n_new=3) == ext[:3]                # 前缀稳定（S9 = x00–x02）

    def test_low_combo_dataset_repeats_once(self):
        e, reg = _registry(n_classes=4, rng=(2, 4), planned=6)                    # 11 组合，原 6 用掉 → 剩 5
        y = _y(4)
        _eff, orig = slice_source_tasks(e, reg, y)
        ext = slice_extension_tasks(e, reg, y, n_new=6)
        orig_sets = {frozenset(t.classes) for t in orig}
        fresh = [frozenset(t.classes) for t in ext if frozenset(t.classes) not in orig_sets]
        assert len(set(fresh)) == 5 and len(ext) == 6


class TestPointsAndPlan:
    def test_rows_for_point_and_test_side(self):
        g = _g74()
        orig = [{"task_id": f"{d}_{i:02d}", "dataset": d, "cls_key": "k", "iters": 60, "origin": ORIGIN_SHARED, "r_prime": 0.5}
                for d in ("A", "B") for i in range(6)]
        ext = [{"task_id": f"{d}_x{i:02d}", "dataset": d, "cls_key": "k", "iters": 60, "origin": ORIGIN_SHARED, "r_prime": 0.5}
               for d in ("A", "B") for i in range(6)]
        assert g.S_POINTS == (6, 9, 12)
        r6, r9, r12 = (g.rows_for_point(orig, ext, s) for s in (6, 9, 12))
        assert r6 == orig
        assert {r["task_id"] for r in r9} == {r["task_id"] for r in orig} | {f"{d}_x{i:02d}" for d in ("A", "B") for i in range(3)}
        assert len(r12) == 24
        assert g.test_slices_for(r12, "A") == [f"A_{i:02d}" for i in range(6)]      # 考侧只用原切片

    def test_plan_cells(self):
        g = _g74()
        space = pq.CLSSpace.from_yaml()
        cells = g.ext_plan(["Dset_x00", "Dset_x01"], space=space)
        assert len(cells) == 2 * 21
        keys = [(c.slice_id, c.cls_key) for c in cells]
        assert len(keys) == len(set(keys))
        by = {}
        for c in cells:
            by.setdefault(c.slice_id, []).append(c.origin)
        assert all(v.count(ORIGIN_SHARED) == 10 and v.count(ORIGIN_RANDOM) == 10 and v.count("ggs") == 1 for v in by.values())

    def test_phi_row_from_ext_baseline(self):
        g = _g74()
        rows = [{"task_id": "Dset_x00", "r_prime": v, "n_eval": 100} for v in (0.50, 0.52, 0.48, 0.51, 0.49)]
        bands = g.baseline_bands_ext(rows)
        assert set(bands) == {"Dset_x00"} and abs(bands["Dset_x00"][0] - 0.50) < 1e-9
        X = np.random.default_rng(0).standard_normal((30, 50, 3)).astype(np.float32)
        row = g.phi_row("Dset_x00", "Dset", X, k=3, n_train=30, band=bands["Dset_x00"])
        assert all(c in row for c in m33.PHI_COLS) and row["base_mean"] == pytest.approx(0.50)


class TestVerdict:
    def test_codes(self):
        g = _g74()
        assert g.verdict_sx({"G-S1": True, "G-S2": True}, "RISING") == "SLICE-WIN"
        assert g.verdict_sx({"G-S1": True, "G-S2": False}, "FLAT") == "SLICE-BLIND"
        assert g.verdict_sx({"G-S1": False, "G-S2": True}, "RISING") == "SLICE-RISING"
        assert g.verdict_sx({"G-S1": False, "G-S2": False}, "FLAT") == "SLICE-FLAT"
        assert g.gate_s1({"n": 756, "mean_a": 0.0242, "mean_b": 0.0215, "p": 0.01}) is True     # a = S6, b = S12
        assert g.gate_s1({"n": 756, "mean_a": 0.0242, "mean_b": 0.0230, "p": 0.01}) is False    # 差 < 0.002
        assert g.gate_s1({"n": 756, "mean_a": 0.0242, "mean_b": 0.0200, "p": 0.2}) is False
