"""源数据切片造任务（TECHNICAL_DESIGN.md 模块 7 步骤 1，对应 AutoCTS++ 3.2.4）。

路线 B（HAR 内 leave-out 验证 comparator 机制）的两级划分：
1. `split_pools`：按固定种子把样本池**分层**划成【源池】/【留出池】，样本零重叠。
   源池 → 切"源任务"（fit 共享 TS2Vec + 造 T-CLSC 训练 label）；
   留出池 → 切"留出任务"（仅 M3 测试 comparator）。
2. `slice_pool_tasks`：在池内切任务。每个任务 = (类别子集, 通道子集, 样本索引)。
   类别子集是制造多样性的主力维度（task_embed 定稿实验：类别子集间余弦 ≈0.65
   区分度最强，通道子集间 ≈0.92 较弱），同池内类别组合互不重复；通道子集辅助
   （单变量共享 TS2Vec 原生支持任意通道数，无需零填充）。长度维不切（HAR 定长
   128；可选扩展：对样本做重采样/裁剪引入长度多样性，本阶段不做）。

与 AutoCTS++ 3.2.4 的差异：原文对预测任务"时间连续切片 + 随机采样通道"；分类
任务无时间轴上的任务边界，改在"类别 × 通道 × 样本子集"维度切，语义对应
"从源数据造出分布互异的子任务"。

泄漏边界（铁律 6）：HAR 内 leave-out 极易让留出样本混进源池——本模块在接口上
强制分离：`make_leaveout_split` 返回的 `source_indices` 是 fit_shared_ts2vec
唯一合法的数据来源，并内置断言两池及其任务样本索引零交集
（`assert_tasks_disjoint`）。

batch 口径：任务样本数下限 `min_samples` 默认 128 = STD_BATCH_SIZE
（configs/train.yaml），保证 cl_trainer 至少能形成一个满 batch；更小的尾 batch
沿用 cl_trainer 已有的 <2 跳过逻辑。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

import numpy as np

POOL_SOURCE = "source"
POOL_HOLDOUT = "holdout"


@dataclass(frozen=True)
class SlicedTask:
    """一个切片任务的完整定义（可 JSON 序列化、逐字段可比 → 可复现性可断言）。

    sample_indices 是**原始数据集全局索引**（升序 tuple），配合 materialize_task
    取数据；classes/channels 均为升序 tuple。pool 标记所属池，防源/留出混用。
    """

    task_id: str
    pool: str
    classes: tuple[int, ...]
    channels: tuple[int, ...]
    sample_indices: tuple[int, ...]


@dataclass(frozen=True)
class LeaveOutSplit:
    """两级划分结果。source_indices 是 fit_shared_ts2vec 唯一合法数据来源（铁律 6）。"""

    source_indices: np.ndarray = field(compare=False)
    holdout_indices: np.ndarray = field(compare=False)
    source_tasks: tuple[SlicedTask, ...]
    holdout_tasks: tuple[SlicedTask, ...]

    def __eq__(self, other):  # ndarray 无法直接参与 dataclass 相等，逐项比
        if not isinstance(other, LeaveOutSplit):
            return NotImplemented
        return (
            np.array_equal(self.source_indices, other.source_indices)
            and np.array_equal(self.holdout_indices, other.holdout_indices)
            and self.source_tasks == other.source_tasks
            and self.holdout_tasks == other.holdout_tasks
        )


def split_pools(
    y: np.ndarray, *, holdout_ratio: float = 0.3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """按类分层把样本索引划成（源池, 留出池），零重叠且完整覆盖。

    分层保证两池都含全部类别（类别子集切片的前提）。返回升序全局索引。
    """
    rng = np.random.default_rng(seed)
    src_parts, hold_parts = [], []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_hold = int(round(len(idx) * holdout_ratio))
        hold_parts.append(idx[:n_hold])
        src_parts.append(idx[n_hold:])
    source = np.sort(np.concatenate(src_parts))
    holdout = np.sort(np.concatenate(hold_parts))
    assert len(np.intersect1d(source, holdout)) == 0
    return source, holdout


def slice_pool_tasks(
    y: np.ndarray,
    pool_indices: np.ndarray,
    *,
    pool_name: str,
    n_tasks: int,
    n_classes_range: tuple[int, int] = (3, 5),
    n_channels_range: tuple[int, int] = (4, 9),
    n_total_channels: int = 9,
    n_samples: int = 1024,
    min_samples: int = 128,   # STD_BATCH_SIZE：保证 cl_trainer 至少一个满 batch
    seed: int = 0,
) -> list[SlicedTask]:
    """在一个池内切 n_tasks 个任务（类别子集互不重复；样本只从 pool_indices 采）。

    每任务：随机类别子集（主力多样性维度）× 随机通道子集 × 随机采
    min(n_samples, 可用数) 个样本；可用数 < min_samples 报错。固定 seed 可复现。
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y[pool_indices])
    lo, hi = n_classes_range
    n_subsets = sum(comb(len(classes), k) for k in range(lo, hi + 1))
    if n_tasks > n_subsets:
        raise ValueError(
            f"n_tasks={n_tasks} 超过可用的不重复类别组合数 {n_subsets}"
            f"（{len(classes)} 类取 {lo}~{hi}）"
        )

    tasks: list[SlicedTask] = []
    seen_subsets: set[frozenset] = set()
    while len(tasks) < n_tasks:
        k = int(rng.integers(lo, hi + 1))
        cls_subset = tuple(sorted(rng.choice(classes, size=k, replace=False).tolist()))
        if frozenset(cls_subset) in seen_subsets:
            continue  # 类别组合互不重复（多样性主力维度）；上限已在前面保证可行
        seen_subsets.add(frozenset(cls_subset))

        kc = int(rng.integers(n_channels_range[0], n_channels_range[1] + 1))
        channels = tuple(sorted(rng.choice(n_total_channels, size=kc, replace=False).tolist()))

        avail = pool_indices[np.isin(y[pool_indices], cls_subset)]
        if len(avail) < min_samples:
            raise ValueError(
                f"池 {pool_name!r} 中类别子集 {cls_subset} 仅 {len(avail)} 样本 "
                f"< min_samples={min_samples}"
            )
        m = min(n_samples, len(avail))
        sample_idx = tuple(np.sort(rng.choice(avail, size=m, replace=False)).tolist())

        tasks.append(SlicedTask(
            task_id=f"{pool_name}_{len(tasks):02d}",
            pool=pool_name,
            classes=cls_subset,
            channels=channels,
            sample_indices=sample_idx,
        ))
    return tasks


def make_leaveout_split(
    y: np.ndarray,
    *,
    n_source_tasks: int,
    n_holdout_tasks: int,
    holdout_ratio: float = 0.3,
    seed: int = 0,
    **task_kw,
) -> LeaveOutSplit:
    """两级划分一步到位：分池 → 各池切任务 → 断言零重叠（铁律 6 的接口防护）。

    fit_shared_ts2vec 只能吃 `X[split.source_indices]`；留出任务仅供 M3 测试
    comparator。task_kw 透传给 slice_pool_tasks（两池同参）。
    """
    source_idx, holdout_idx = split_pools(y, holdout_ratio=holdout_ratio, seed=seed)
    # 两池用不同派生种子，避免"同 seed 同序采样"造成源/留出任务结构镜像
    source_tasks = slice_pool_tasks(
        y, source_idx, pool_name=POOL_SOURCE, n_tasks=n_source_tasks,
        seed=seed, **task_kw,
    )
    holdout_tasks = slice_pool_tasks(
        y, holdout_idx, pool_name=POOL_HOLDOUT, n_tasks=n_holdout_tasks,
        seed=seed + 1_000_003, **task_kw,
    )
    assert_tasks_disjoint(source_tasks, holdout_tasks)
    return LeaveOutSplit(
        source_indices=source_idx,
        holdout_indices=holdout_idx,
        source_tasks=tuple(source_tasks),
        holdout_tasks=tuple(holdout_tasks),
    )


def materialize_task(
    task: SlicedTask, X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """按任务定义取数据：X (n, T, |channels|) float32；标签重映射到 0..k-1。

    重映射保序（classes 升序第 i 类 → i）：下游分类头要求连续标签。
    """
    idx = np.array(task.sample_indices)
    Xt = X[idx][:, :, list(task.channels)].astype(np.float32, copy=False)
    remap = {orig: local for local, orig in enumerate(task.classes)}
    yt = np.array([remap[v] for v in y[idx]], dtype=np.int64)
    return Xt, yt


def assert_tasks_disjoint(
    source_tasks: list[SlicedTask] | tuple[SlicedTask, ...],
    holdout_tasks: list[SlicedTask] | tuple[SlicedTask, ...],
) -> None:
    """泄漏自检（铁律 6）：任一源任务样本不得出现在任一留出任务中。"""
    src_used = set().union(*(t.sample_indices for t in source_tasks)) if source_tasks else set()
    hold_used = set().union(*(t.sample_indices for t in holdout_tasks)) if holdout_tasks else set()
    overlap = src_used & hold_used
    assert not overlap, (
        f"源任务与留出任务样本索引重叠 {len(overlap)} 个（如 {sorted(overlap)[:5]}）"
        "——违反铁律 6 泄漏边界"
    )
