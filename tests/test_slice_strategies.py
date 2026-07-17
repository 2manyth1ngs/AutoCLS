"""切片策略单测（可插拔策略架构：核心 task_slicer 与任务类型解耦）。

架构约定：
- 核心 `task_slicer.slice_pool_tasks` 只负责通用逻辑（通道/样本采样、min_samples、
  eligible⊆pool 泄漏复核、任务记录），"怎么选类别子集"下放到 SliceStrategy；
- `RandomClassSubsetSlicer`：通用分类策略（原核心内置逻辑迁出，默认 (2,4)）；
- `DynamicStaticStratifiedSlicer`：**HAR 分类专用**，按动/静类别构成切三档
  （纯动/纯静/混合）主动制造任务多样性——不适用于预测/异常检测；
- 扩展点：将来预测任务新增自己的策略（如按数据集+horizon 产出 TaskSpec），
  核心与分类策略零改动。

合成标签沿用 test_task_slicer 口径：6 类，{0,1,2} 视作"动态"、{3,4,5} 视作"静态"。
"""
import numpy as np
import pytest

from autocls.data.slice_strategies import (
    DynamicStaticStratifiedSlicer,
    RandomClassSubsetSlicer,
    resolve_slice_strategy,
)
from autocls.task_types import TASK_CLASSIFICATION, TASK_FORECASTING
from autocls.data.task_slicer import (
    TaskSpec,
    make_leaveout_split,
    slice_pool_tasks,
    split_pools,
)

N_CLASSES = 6
DYNAMIC = (0, 1, 2)
STATIC = (3, 4, 5)


def _toy_labels(n=600, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, N_CLASSES, size=n).astype(np.int64)


def _pool(y, seed=0) -> np.ndarray:
    pool, _ = split_pools(y, holdout_ratio=0.3, seed=seed)
    return pool


# ---------------------------------------------------------- RandomClassSubsetSlicer
class TestRandomClassSubsetSlicer:
    def _specs(self, n_tasks=8, seed=0, **kw):
        y = _toy_labels()
        strat = RandomClassSubsetSlicer(**kw)
        return y, _pool(y), strat.propose(y, _pool(y), n_tasks, np.random.default_rng(seed))

    def test_n_specs_distinct_subsets_default_range(self):
        _, _, specs = self._specs(n_tasks=10)
        assert len(specs) == 10
        assert all(isinstance(s, TaskSpec) for s in specs)
        subsets = [frozenset(s.classes) for s in specs]
        assert len(set(subsets)) == len(subsets)  # 类别组合互不重复
        for s in specs:
            assert 2 <= len(s.classes) <= 4  # 默认 n_classes_range=(2,4)（已定决策）
            assert list(s.classes) == sorted(set(s.classes))

    def test_custom_range(self):
        _, _, specs = self._specs(n_tasks=6, n_classes_range=(3, 5))
        assert all(3 <= len(s.classes) <= 5 for s in specs)

    def test_eligible_matches_subset_and_pool(self):
        y, pool, specs = self._specs(n_tasks=8)
        pool_set = set(pool.tolist())
        for s in specs:
            assert set(s.eligible.tolist()) <= pool_set
            assert set(y[s.eligible]) <= set(s.classes)
            # eligible 必须是"池内该类别子集的全部样本"，核心从中采样
            expect = pool[np.isin(y[pool], s.classes)]
            np.testing.assert_array_equal(np.sort(s.eligible), np.sort(expect))

    def test_too_many_subsets_raises(self):
        with pytest.raises(ValueError, match="类别组合"):
            self._specs(n_tasks=2, n_classes_range=(6, 6))

    def test_allow_repeat_exceeds_unique_combos(self):
        # 路线 A 低类数数据集（如 2 类 EEG）：类别组合不够 n_tasks 时，允许重复类别
        # 子集，多样性由核心的通道/样本子集承担（核心已支持，见 test_custom_dummy_strategy）。
        # 2 类只有 1 个 (2,2) 子集，但要 4 个任务 → allow_repeat 不再报错。
        y = np.tile([0, 1], 300).astype(np.int64)
        strat = RandomClassSubsetSlicer(n_classes_range=(2, 2), allow_repeat=True)
        specs = strat.propose(y, _pool(y), 4, np.random.default_rng(0))
        assert len(specs) == 4
        assert all(s.classes == (0, 1) for s in specs)

    def test_allow_repeat_prefers_unique_first(self):
        # allow_repeat=True 但组合充裕时，仍先出满不重复子集（最大化类别多样性），
        # 仅在耗尽后才重复。6 类 (2,2)=15 个组合，取 4 个应全不重复。
        _, _, specs = self._specs(n_tasks=4, n_classes_range=(2, 2), allow_repeat=True)
        subsets = [frozenset(s.classes) for s in specs]
        assert len(set(subsets)) == 4  # 未耗尽 → 全不重复

    def test_allow_repeat_false_is_default_and_still_raises(self):
        # 默认 allow_repeat=False：组合不足即报错的护栏语义不变（6 类取 6 只有 1 个组合）。
        with pytest.raises(ValueError, match="类别组合"):
            self._specs(n_tasks=4, n_classes_range=(6, 6), allow_repeat=False)

    def test_reproducible(self):
        _, _, a = self._specs(n_tasks=6, seed=7)
        _, _, b = self._specs(n_tasks=6, seed=7)
        assert [s.classes for s in a] == [s.classes for s in b]
        _, _, c = self._specs(n_tasks=6, seed=8)
        assert [s.classes for s in a] != [s.classes for s in c]


# ---------------------------------------------------- DynamicStaticStratifiedSlicer
class TestDynamicStaticStratifiedSlicer:
    def _specs(self, n_tasks=16, seed=0, **kw):
        y = _toy_labels()
        strat = DynamicStaticStratifiedSlicer(
            dynamic_classes=DYNAMIC, static_classes=STATIC, **kw
        )
        return y, strat.propose(y, _pool(y), n_tasks, np.random.default_rng(seed))

    def test_three_tiers_capped_allocation(self):
        # 16 任务：目标 ≈5/5/6，但纯档不重复子集上限 C(3,2)+C(3,3)=4，
        # 超额转入混合档 → 4/4/8
        _, specs = self._specs(n_tasks=16)
        tags = [s.tag for s in specs]
        assert tags.count(DynamicStaticStratifiedSlicer.TAG_DYNAMIC) == 4
        assert tags.count(DynamicStaticStratifiedSlicer.TAG_STATIC) == 4
        assert tags.count(DynamicStaticStratifiedSlicer.TAG_MIXED) == 8

    def test_balanced_allocation_when_uncapped(self):
        _, specs = self._specs(n_tasks=9)
        tags = [s.tag for s in specs]
        assert tags.count(DynamicStaticStratifiedSlicer.TAG_DYNAMIC) == 3
        assert tags.count(DynamicStaticStratifiedSlicer.TAG_STATIC) == 3
        assert tags.count(DynamicStaticStratifiedSlicer.TAG_MIXED) == 3

    def test_pure_tiers_class_membership(self):
        _, specs = self._specs(n_tasks=16)
        for s in specs:
            if s.tag == DynamicStaticStratifiedSlicer.TAG_DYNAMIC:
                assert set(s.classes) <= set(DYNAMIC)  # 动态占比=1.0
            elif s.tag == DynamicStaticStratifiedSlicer.TAG_STATIC:
                assert set(s.classes) <= set(STATIC)   # 动态占比=0.0

    def test_mixed_truly_mixed(self):
        # 混合档必须真的混合：动/静各 ≥1，不许退化成纯档
        _, specs = self._specs(n_tasks=16)
        mixed = [s for s in specs if s.tag == DynamicStaticStratifiedSlicer.TAG_MIXED]
        assert mixed
        for s in mixed:
            assert set(s.classes) & set(DYNAMIC)
            assert set(s.classes) & set(STATIC)

    def test_subsets_distinct_and_sizes_in_range(self):
        _, specs = self._specs(n_tasks=16)
        subsets = [frozenset(s.classes) for s in specs]
        assert len(set(subsets)) == len(subsets)
        assert all(2 <= len(s.classes) <= 4 for s in specs)  # 默认 (2,4)

    def test_eligible_matches_subset_and_pool(self):
        y, specs = self._specs(n_tasks=16)
        pool = _pool(y)
        for s in specs:
            expect = pool[np.isin(y[pool], s.classes)]
            np.testing.assert_array_equal(np.sort(s.eligible), np.sort(expect))

    def test_infeasible_n_tasks_raises(self):
        # 6 类 (2,4) 总容量 = 4 + 4 + 42 = 50
        with pytest.raises(ValueError, match="类别组合"):
            self._specs(n_tasks=51)
        _, specs = self._specs(n_tasks=50)  # 恰好用尽
        assert len(specs) == 50

    def test_missing_declared_class_raises(self):
        y = np.array([0, 1, 2, 3] * 50, dtype=np.int64)  # 池中无类 4/5
        strat = DynamicStaticStratifiedSlicer(dynamic_classes=DYNAMIC, static_classes=STATIC)
        with pytest.raises(ValueError, match="类别"):
            strat.propose(y, np.arange(len(y)), 6, np.random.default_rng(0))

    def test_overlapping_groups_raise(self):
        with pytest.raises(ValueError, match="交集"):
            DynamicStaticStratifiedSlicer(dynamic_classes=(0, 1, 2), static_classes=(2, 3))

    def test_reproducible(self):
        _, a = self._specs(n_tasks=12, seed=7)
        _, b = self._specs(n_tasks=12, seed=7)
        assert [(s.classes, s.tag) for s in a] == [(s.classes, s.tag) for s in b]


# ------------------------------------------------- task_type → 策略分发（铁律 7）
class TestResolveSliceStrategy:
    def test_classification_resolves_to_stratified(self):
        # 阶段一仅 HAR：classification 默认 = 动/静三档分层（已拍板）
        strat = resolve_slice_strategy(TASK_CLASSIFICATION)
        assert isinstance(strat, DynamicStaticStratifiedSlicer)

    def test_forecasting_is_stub(self):
        # 扩展点：预测接入时新增策略类（按数据集+horizon）并在分发点加 case
        with pytest.raises(NotImplementedError, match="切片策略"):
            resolve_slice_strategy(TASK_FORECASTING)

    def test_unknown_task_type_rejected(self):
        with pytest.raises(ValueError, match="task_type"):
            resolve_slice_strategy("segmentation")


# ------------------------------------------------------- 核心可插拔性（解耦自检）
TASK_KW = dict(n_samples=64, min_samples=16)


class TestCorePluggability:
    def test_stratified_strategy_via_core(self):
        # 策略经接口注入核心；tag 透传到 SlicedTask；采样约束不变
        y = _toy_labels()
        pool = _pool(y)
        tasks = slice_pool_tasks(
            y, pool, pool_name="source", n_tasks=12, seed=0,
            strategy=DynamicStaticStratifiedSlicer(
                dynamic_classes=DYNAMIC, static_classes=STATIC),
            **TASK_KW,
        )
        assert len(tasks) == 12
        tags = {t.tag for t in tasks}
        assert tags == {
            DynamicStaticStratifiedSlicer.TAG_DYNAMIC,
            DynamicStaticStratifiedSlicer.TAG_STATIC,
            DynamicStaticStratifiedSlicer.TAG_MIXED,
        }
        pool_set = set(pool.tolist())
        for t in tasks:
            assert set(t.sample_indices) <= pool_set
            assert set(y[list(t.sample_indices)]) <= set(t.classes)
            assert len(t.sample_indices) >= TASK_KW["min_samples"]

    def test_custom_dummy_strategy(self):
        # 解耦自检：任意自定义策略可替换——核心不含任何"怎么选类别"的硬编码
        y = _toy_labels()
        pool = _pool(y)

        class FixedSubsetStrategy:
            """极简自定义策略：所有任务同一类别子集（仅测接口，非真实用法）。"""

            def propose(self, y, pool_indices, n_tasks, rng):
                eligible = pool_indices[np.isin(y[pool_indices], (0, 3))]
                return [
                    TaskSpec(classes=(0, 3), eligible=eligible, tag=f"fixed_{i}")
                    for i in range(n_tasks)
                ]

        tasks = slice_pool_tasks(
            y, pool, pool_name="source", n_tasks=3, seed=0,
            strategy=FixedSubsetStrategy(), **TASK_KW,
        )
        assert [t.tag for t in tasks] == ["fixed_0", "fixed_1", "fixed_2"]
        assert all(t.classes == (0, 3) for t in tasks)

    def test_core_rejects_out_of_pool_eligible(self):
        # 泄漏防护在核心复核：策略产出越池索引必须被拦截（铁律 6 不因策略而回退）
        y = _toy_labels()
        pool = _pool(y)
        out_of_pool = np.setdiff1d(np.arange(len(y)), pool)[:32]

        class LeakyStrategy:
            def propose(self, y, pool_indices, n_tasks, rng):
                return [TaskSpec(classes=(0, 1), eligible=out_of_pool, tag="leak")]

        with pytest.raises(ValueError, match="越池"):
            slice_pool_tasks(
                y, pool, pool_name="source", n_tasks=1, seed=0,
                strategy=LeakyStrategy(), **TASK_KW,
            )

    def test_make_leaveout_split_with_stratified_strategy(self):
        # 要求三：换策略后两级划分的泄漏防护完整保留
        sp = make_leaveout_split(
            _toy_labels(), n_source_tasks=9, n_holdout_tasks=6,
            holdout_ratio=0.3, seed=0,
            strategy=DynamicStaticStratifiedSlicer(
                dynamic_classes=DYNAMIC, static_classes=STATIC),
            **TASK_KW,
        )
        src_used = set().union(*(t.sample_indices for t in sp.source_tasks))
        hold_used = set().union(*(t.sample_indices for t in sp.holdout_tasks))
        assert src_used & hold_used == set()
        assert src_used <= set(sp.source_indices.tolist())
        assert hold_used <= set(sp.holdout_indices.tolist())
        assert {t.tag for t in sp.source_tasks} == {
            DynamicStaticStratifiedSlicer.TAG_DYNAMIC,
            DynamicStaticStratifiedSlicer.TAG_STATIC,
            DynamicStaticStratifiedSlicer.TAG_MIXED,
        }
