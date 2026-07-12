"""task_slicer 单测（TECHNICAL_DESIGN.md 模块 7 步骤 1；路线 B：HAR 内 leave-out）。

两级划分语义：
1. 先按固定种子把样本池分层划成【源池】/【留出池】（样本零重叠）；
2. 再在各自池内切任务——类别子集是制造多样性的主力维度（task_embed 定稿
   实验：类别维区分度 0.65 > 通道维 0.92），通道子集辅助。

泄漏边界（铁律 6）：源任务与留出任务来自不重叠的池，接口上强制分离；
fit_shared_ts2vec 只能吃源池数据。全部用合成标签/数据，CPU 秒级跑完；
真实 HAR 上的任务 E 多样性验证见 scripts/sanity_task_slicer.py。
"""
import numpy as np
import pytest

from autocls.data.slice_strategies import RandomClassSubsetSlicer
from autocls.task_types import TASK_CLASSIFICATION, TASK_FORECASTING
from autocls.data.task_slicer import (
    SlicedTask,
    assert_tasks_disjoint,
    make_leaveout_split,
    materialize_task,
    slice_pool_tasks,
    split_pools,
)

N_CLASSES = 6


def _toy_labels(n=600, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, N_CLASSES, size=n).astype(np.int64)


def _toy_X(n=600, t=16, c=9, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, t, c)).astype(np.float32)


# 合成池够小，压低采样下限让 3 类子集也能成任务
TASK_KW = dict(n_samples=64, min_samples=16)


# ---------------------------------------------------------------- split_pools
class TestSplitPools:
    def test_disjoint_and_complete_partition(self):
        y = _toy_labels()
        src, hold = split_pools(y, holdout_ratio=0.3, seed=0)
        assert len(np.intersect1d(src, hold)) == 0  # 铁律 6：两池样本零重叠
        assert len(src) + len(hold) == len(y)
        np.testing.assert_array_equal(np.sort(np.concatenate([src, hold])), np.arange(len(y)))

    def test_holdout_ratio(self):
        y = _toy_labels()
        _, hold = split_pools(y, holdout_ratio=0.3, seed=0)
        assert abs(len(hold) / len(y) - 0.3) < 0.02

    def test_stratified_both_pools_have_all_classes(self):
        # 分层划分：类别子集切片要求两池都有全部 6 类可选
        y = _toy_labels()
        src, hold = split_pools(y, holdout_ratio=0.3, seed=0)
        assert set(y[src]) == set(range(N_CLASSES))
        assert set(y[hold]) == set(range(N_CLASSES))

    def test_reproducible(self):
        y = _toy_labels()
        a = split_pools(y, holdout_ratio=0.3, seed=42)
        b = split_pools(y, holdout_ratio=0.3, seed=42)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        c = split_pools(y, holdout_ratio=0.3, seed=43)
        assert not np.array_equal(a[0], c[0])


# ---------------------------------------------------------------- slice_pool_tasks
class TestSlicePoolTasks:
    def _tasks(self, n_tasks=8, seed=0, **kw):
        y = _toy_labels()
        pool, _ = split_pools(y, holdout_ratio=0.3, seed=0)
        return slice_pool_tasks(
            y, pool, pool_name="source", n_tasks=n_tasks, seed=seed,
            **{**TASK_KW, **kw},
        )

    def test_returns_n_tasks_with_pool_tag(self):
        tasks = self._tasks(n_tasks=8)
        assert len(tasks) == 8
        assert all(isinstance(t, SlicedTask) for t in tasks)
        assert all(t.pool == "source" for t in tasks)
        assert len({t.task_id for t in tasks}) == 8  # id 唯一

    def test_sample_indices_within_pool(self):
        y = _toy_labels()
        pool, _ = split_pools(y, holdout_ratio=0.3, seed=0)
        tasks = slice_pool_tasks(
            y, pool, pool_name="source", n_tasks=8, seed=0, **TASK_KW
        )
        pool_set = set(pool.tolist())
        for t in tasks:
            assert set(t.sample_indices) <= pool_set

    def test_labels_match_class_subset(self):
        y = _toy_labels()
        for t in self._tasks(n_tasks=8):
            assert set(y[list(t.sample_indices)]) <= set(t.classes)

    def test_class_subsets_distinct_across_tasks(self):
        # 多样性主力维度：不同任务必须取不同类别组合
        tasks = self._tasks(n_tasks=10)
        subsets = [frozenset(t.classes) for t in tasks]
        assert len(set(subsets)) == len(subsets)

    def test_subset_sizes_default_range(self):
        # 缺省策略经 task_type 分发（classification → 动/静分层），range 恒 (2,4)（已定决策）
        for t in self._tasks(n_tasks=10, n_channels_range=(4, 9)):
            assert 2 <= len(t.classes) <= 4
            assert 4 <= len(t.channels) <= 9
            assert list(t.classes) == sorted(set(t.classes))
            assert list(t.channels) == sorted(set(t.channels))
            assert all(0 <= c < 9 for c in t.channels)

    def test_strategy_range_override(self):
        # n_classes_range 属于策略、不属于核心：经 strategy 参数注入
        tasks = self._tasks(
            n_tasks=10, strategy=RandomClassSubsetSlicer(n_classes_range=(3, 5))
        )
        assert all(3 <= len(t.classes) <= 5 for t in tasks)

    def test_min_samples_enforced(self):
        for t in self._tasks(n_tasks=8):
            assert len(t.sample_indices) >= TASK_KW["min_samples"]

    def test_insufficient_pool_raises(self):
        y = _toy_labels(n=60)  # 每类 ~10 样本，3 类子集 ~30 < min_samples=50 必然触发
        pool = np.arange(60)
        with pytest.raises(ValueError, match="min_samples"):
            slice_pool_tasks(
                y, pool, pool_name="source", n_tasks=2, seed=0,
                strategy=RandomClassSubsetSlicer(n_classes_range=(3, 3)),
                n_samples=64, min_samples=50,
            )

    def test_too_many_distinct_subsets_raises(self):
        # (6,6) 只有 1 种类别组合，要 2 个任务必然重复 → 策略显式报错
        with pytest.raises(ValueError, match="类别组合"):
            self._tasks(n_tasks=2, strategy=RandomClassSubsetSlicer(n_classes_range=(6, 6)))

    def test_reproducible_same_seed(self):
        a = self._tasks(n_tasks=6, seed=7)
        b = self._tasks(n_tasks=6, seed=7)
        assert a == b  # SlicedTask 为 frozen dataclass，逐字段相等
        c = self._tasks(n_tasks=6, seed=8)
        assert a != c


# ---------------------------------------------------------------- make_leaveout_split
class TestMakeLeaveoutSplit:
    def _split(self, seed=0):
        return make_leaveout_split(
            _toy_labels(), n_source_tasks=8, n_holdout_tasks=4,
            holdout_ratio=0.3, seed=seed, **TASK_KW,
        )

    def test_source_holdout_zero_overlap(self):
        # 泄漏自检核心：任一源任务样本绝不出现在任一留出任务中
        sp = self._split()
        src_used = set().union(*(t.sample_indices for t in sp.source_tasks))
        hold_used = set().union(*(t.sample_indices for t in sp.holdout_tasks))
        assert src_used & hold_used == set()
        assert src_used <= set(sp.source_indices.tolist())
        assert hold_used <= set(sp.holdout_indices.tolist())

    def test_pool_tags_and_counts(self):
        sp = self._split()
        assert len(sp.source_tasks) == 8 and len(sp.holdout_tasks) == 4
        assert all(t.pool == "source" for t in sp.source_tasks)
        assert all(t.pool == "holdout" for t in sp.holdout_tasks)

    def test_source_indices_feed_ts2vec(self):
        # fit_shared_ts2vec(X[sp.source_indices]) 是唯一合法用法：
        # source_indices 与留出池必须零交集
        sp = self._split()
        assert len(np.intersect1d(sp.source_indices, sp.holdout_indices)) == 0

    def test_reproducible(self):
        assert self._split(seed=3) == self._split(seed=3)


# ---------------------------------------------------------------- materialize_task
class TestMaterializeTask:
    def test_shapes_and_label_remap(self):
        X, y = _toy_X(), _toy_labels()
        sp = make_leaveout_split(
            y, n_source_tasks=4, n_holdout_tasks=2, holdout_ratio=0.3, seed=0, **TASK_KW
        )
        t = sp.source_tasks[0]
        Xt, yt = materialize_task(t, X, y)
        assert Xt.shape == (len(t.sample_indices), X.shape[1], len(t.channels))
        assert Xt.dtype == np.float32
        # 标签重映射到 0..k-1（下游分类头要求连续标签）
        assert set(yt) <= set(range(len(t.classes)))
        # 映射保序：classes 排序后第 i 类 → i
        for local, orig in enumerate(t.classes):
            mask = y[list(t.sample_indices)] == orig
            assert (yt[mask] == local).all()

    def test_channel_subset_content(self):
        X, y = _toy_X(), _toy_labels()
        sp = make_leaveout_split(
            y, n_source_tasks=4, n_holdout_tasks=2, holdout_ratio=0.3, seed=0, **TASK_KW
        )
        t = sp.source_tasks[0]
        Xt, _ = materialize_task(t, X, y)
        np.testing.assert_array_equal(
            Xt, X[np.array(t.sample_indices)][:, :, list(t.channels)]
        )


# ---------------------------------------------------------------- task_type 分发（铁律 7）
class TestTaskTypeDispatch:
    def _pool(self, y):
        pool, _ = split_pools(y, holdout_ratio=0.3, seed=0)
        return pool

    def test_task_type_recorded_default_classification(self):
        # task 数据结构必须带显式 task_type（当前恒 classification）
        y = _toy_labels()
        tasks = slice_pool_tasks(
            y, self._pool(y), pool_name="source", n_tasks=6, seed=0, **TASK_KW
        )
        assert all(t.task_type == TASK_CLASSIFICATION for t in tasks)

    def test_task_type_flows_through_leaveout_split(self):
        sp = make_leaveout_split(
            _toy_labels(), n_source_tasks=6, n_holdout_tasks=3,
            holdout_ratio=0.3, seed=0, **TASK_KW,
        )
        for t in sp.source_tasks + sp.holdout_tasks:
            assert t.task_type == TASK_CLASSIFICATION

    def test_unknown_task_type_rejected(self):
        y = _toy_labels()
        with pytest.raises(ValueError, match="task_type"):
            slice_pool_tasks(
                y, self._pool(y), pool_name="source", n_tasks=4, seed=0,
                task_type="segmentation", **TASK_KW,
            )

    def test_forecasting_dispatch_is_stub(self):
        # 扩展点：forecasting 尚无切片策略，分发点显式拒绝而非隐式当分类处理
        y = _toy_labels()
        with pytest.raises(NotImplementedError, match="切片策略"):
            slice_pool_tasks(
                y, self._pool(y), pool_name="source", n_tasks=4, seed=0,
                task_type=TASK_FORECASTING, **TASK_KW,
            )

    def test_explicit_strategy_still_stamps_task_type(self):
        # 显式传策略时不走分发，但 task_type 仍必须记录在任务上
        y = _toy_labels()
        tasks = slice_pool_tasks(
            y, self._pool(y), pool_name="source", n_tasks=6, seed=0,
            strategy=RandomClassSubsetSlicer(), **TASK_KW,
        )
        assert all(t.task_type == TASK_CLASSIFICATION for t in tasks)


# ---------------------------------------------------------------- 泄漏守卫
class TestAssertTasksDisjoint:
    def _task(self, pool, indices):
        return SlicedTask(
            task_id=f"{pool}_x", pool=pool, classes=(0, 1, 2),
            channels=(0, 1, 2, 3), sample_indices=tuple(indices),
        )

    def test_passes_when_disjoint(self):
        assert_tasks_disjoint([self._task("source", [0, 1])], [self._task("holdout", [2, 3])])

    def test_raises_on_overlap(self):
        with pytest.raises(AssertionError, match="重叠"):
            assert_tasks_disjoint(
                [self._task("source", [0, 1])], [self._task("holdout", [1, 2])]
            )
