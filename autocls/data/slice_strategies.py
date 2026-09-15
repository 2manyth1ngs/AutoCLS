"""分类任务的切片策略（可插拔；接口契约见 task_slicer.SliceStrategy / TaskSpec）。

核心/策略分工：task_slicer 核心只做通用逻辑（池划分、样本/通道采样、泄漏防护、
任务记录）；"每个任务选哪些类别、哪些样本可采"这类**任务类型相关**的逻辑全部
在策略里。本文件只放分类策略：

- `RandomClassSubsetSlicer`：通用分类策略。随机不重复类别子集（原 task_slicer
  内置逻辑迁出），适用任何多类分类数据集。
- `DynamicStaticStratifiedSlicer`：**HAR 分类专用**。按动/静类别构成把任务分
  三档（纯动/纯静/混合）主动往极端推制造多样性。前提是类别有"动/静"语义分组
  （HAR：{0,1,2}=走/上楼/下楼，{3,4,5}=坐/站/躺），**不适用于预测/异常检测**，
  也不适用于无此语义分组的分类数据集（那种情况用 RandomClassSubsetSlicer）。

扩展点（新任务类型如何接入）：
- 策略只需实现 `propose(y, pool_indices, n_tasks, rng) -> list[TaskSpec]`，
  鸭子类型即可，无需继承。
- 将来预测/异常检测各建自己的策略，例如预测任务的 ForecastHorizonSlicer 可产出
  `TaskSpec(classes=(), eligible=时间连续窗口的起点索引, tag=f"horizon={h}")`
  （y 传形状占位即可，预测无类别标签）；若需记录 horizon 等新字段，给
  SlicedTask 加带默认值的可选字段——task_slicer 核心与本文件的分类策略零改动。
- 策略必须无状态（propose 是 (y, pool_indices, n_tasks, rng) 的纯函数）：
  make_leaveout_split 用同一策略实例在源/留出两池上各调一次。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np

from ..task_types import TASK_CLASSIFICATION, validate_task_type
from .task_slicer import SliceStrategy, TaskSpec


def resolve_slice_strategy(task_type: str) -> SliceStrategy:
    """按 task_type 分发默认切片策略（铁律 7 在 task_slicer 的分发点）。

    classification → DynamicStaticStratifiedSlicer（阶段一只有 HAR，动/静三档
    分层已拍板为源任务默认；其他分类数据集若无动/静语义分组，应显式传
    RandomClassSubsetSlicer，或将来把本分发细化到数据集维度）。
    扩展点：forecasting 接入时新增其策略类（如按数据集+horizon 产出 TaskSpec
    的 ForecastHorizonSlicer）并在此加 case——核心与既有分类策略零改动。
    """
    validate_task_type(task_type)
    if task_type == TASK_CLASSIFICATION:
        return DynamicStaticStratifiedSlicer()
    raise NotImplementedError(
        f"task_type={task_type!r} 的切片策略未实现——接入时新增策略类并在"
        "resolve_slice_strategy 加 case（扩展点见模块 docstring）"
    )


def _eligible(y: np.ndarray, pool_indices: np.ndarray, subset: tuple[int, ...]) -> np.ndarray:
    """池内属于类别子集的全部样本索引（核心从中采样，保证不越池）。"""
    return pool_indices[np.isin(y[pool_indices], subset)]


@dataclass(frozen=True)
class RandomClassSubsetSlicer:
    """随机不重复类别子集策略（通用分类默认）。

    n_classes_range 默认 (2, 4)：多样性探针（results/task_slicer_sanity/probe_cls23）
    表明更小子集分布更"纯"、任务间更易拉开；上限 4 保留任务难度，避免 2 类过简
    导致 R' 贴天花板（M2 已知高分簇内分辨力弱）。

    allow_repeat（路线 A 低类数数据集）：默认 False——类别组合数 < n_tasks 即报错
    （护栏，防误配）。置 True 时先出满不重复子集（最大化类别多样性），耗尽后允许
    重复类别子集，剩余多样性由核心的通道/样本子集承担（核心已支持重复子集，见
    tests/test_slice_strategies.py::test_custom_dummy_strategy）。为 2 类 EEG 源
    （MotorImagery/SCP1/SCP2 仅 1 个 (2,2) 组合却要多任务）而设。
    """

    n_classes_range: tuple[int, int] = (2, 4)
    allow_repeat: bool = False

    def propose(
        self, y: np.ndarray, pool_indices: np.ndarray, n_tasks: int,
        rng: np.random.Generator,
    ) -> list[TaskSpec]:
        classes = np.unique(y[pool_indices])
        lo, hi = self.n_classes_range
        n_subsets = sum(comb(len(classes), k) for k in range(lo, hi + 1))
        if n_tasks > n_subsets and not self.allow_repeat:
            raise ValueError(
                f"n_tasks={n_tasks} 超过可用的不重复类别组合数 {n_subsets}"
                f"（{len(classes)} 类取 {lo}~{hi}）；低类数源池数据集应传 allow_repeat=True"
            )
        specs: list[TaskSpec] = []
        seen: set[frozenset] = set()
        while len(specs) < n_tasks:
            k = int(rng.integers(lo, hi + 1))
            subset = tuple(sorted(rng.choice(classes, size=k, replace=False).tolist()))
            fs = frozenset(subset)
            # 不重复优先：仅当已耗尽全部组合（allow_repeat）才接受重复子集
            if fs in seen and (not self.allow_repeat or len(seen) < n_subsets):
                continue  # 类别组合互不重复是多样性主力维度
            seen.add(fs)
            specs.append(TaskSpec(classes=subset, eligible=_eligible(y, pool_indices, subset)))
        return specs


@dataclass(frozen=True)
class ExcludingClassSubsetSlicer(RandomClassSubsetSlicer):
    """扩展切片用（WP16 SX，results/tclsc_slices/PROTOCOL.md §1.4）：与 RandomClassSubsetSlicer 同一抽样式，
    但把 ``exclude``（既有切片的类别子集）视作"已用过"——同一 RNG 流内先出未用过的组合，**全部组合耗尽后才允许重复**
    （Ethanol / HMD 4 类 11 个组合、原 6 切片用掉 6，第 6 个扩展切片只能重复类别子集，多样性转由通道 / 样本子集承担；
    AutoCTS++ dataset_slice.py 本来就不去重）。产出 tag="ext"。allow_repeat 字段在此无意义（耗尽即重复）。
    """

    exclude: tuple[tuple[int, ...], ...] = ()

    def propose(
        self, y: np.ndarray, pool_indices: np.ndarray, n_tasks: int,
        rng: np.random.Generator,
    ) -> list[TaskSpec]:
        classes = np.unique(y[pool_indices])
        lo, hi = self.n_classes_range
        n_subsets = sum(comb(len(classes), k) for k in range(lo, hi + 1))
        seen: set[frozenset] = {frozenset(c) for c in self.exclude}
        specs: list[TaskSpec] = []
        while len(specs) < n_tasks:
            k = int(rng.integers(lo, hi + 1))
            subset = tuple(sorted(rng.choice(classes, size=k, replace=False).tolist()))
            fs = frozenset(subset)
            if fs in seen and len(seen) < n_subsets:
                continue                        # 还有没用过的组合 → 不接受重复
            seen.add(fs)
            specs.append(TaskSpec(classes=subset, eligible=_eligible(y, pool_indices, subset), tag="ext"))
        return specs


@dataclass(frozen=True)
class DynamicStaticStratifiedSlicer:
    """按动/静类别构成三档分层的策略（**仅适用于 HAR 分类**）。

    动机（task_slicer 多样性诊断）：任务间 TS2Vec 级相似度主要由"动/静类别构成"
    驱动（动态占比之差与相似度 pearson≈-0.87）；随机采类别子集采不到动/静极端对，
    16 任务 off-diag mean=0.973 拉不开。修复思路是**主动往极端推**而不是缩小
    类别数：三档大致均衡——
    - 纯动态档（tag=pure_dynamic）：子集只从 dynamic_classes 取，动态占比=1.0；
    - 纯静态档（tag=pure_static）：子集只从 static_classes 取，动态占比=0.0；
    - 混合档（tag=mixed）：动/静各至少 1 类（占比落中间），保证"真的混合"。
    纯档受不重复子集数上限约束（3 类取 2~3 只有 4 种），超额名额转入混合档：
    如 16 任务目标 ≈5/5/6 → 实际 4/4/8。

    仅当类别有动/静语义分组时有意义；预测/异常检测或无此分组的数据集请换策略
    （扩展点见模块 docstring）。
    """

    dynamic_classes: tuple[int, ...] = (0, 1, 2)  # HAR: 走/上楼/下楼
    static_classes: tuple[int, ...] = (3, 4, 5)   # HAR: 坐/站/躺
    n_classes_range: tuple[int, int] = (2, 4)

    TAG_DYNAMIC = "pure_dynamic"
    TAG_STATIC = "pure_static"
    TAG_MIXED = "mixed"

    def __post_init__(self):
        overlap = set(self.dynamic_classes) & set(self.static_classes)
        if overlap:
            raise ValueError(f"动/静类别集合有交集 {sorted(overlap)}，必须互斥")
        if not self.dynamic_classes or not self.static_classes:
            raise ValueError("动/静类别集合都不能为空")

    # ---- 各档的合法类别组合全枚举（HAR 量级类别数下枚举开销可忽略）----
    def _pure_combos(self, group: tuple[int, ...]) -> list[tuple[int, ...]]:
        lo, hi = self.n_classes_range
        return [
            tuple(c)
            for k in range(lo, min(hi, len(group)) + 1)
            for c in combinations(sorted(group), k)
        ]

    def _mixed_combos(self) -> list[tuple[int, ...]]:
        lo, hi = self.n_classes_range
        dyn, stat = sorted(self.dynamic_classes), sorted(self.static_classes)
        out = []
        for k in range(max(lo, 2), hi + 1):  # 混合至少 2 类才可能动/静各一
            for nd in range(max(1, k - len(stat)), min(len(dyn), k - 1) + 1):
                for a in combinations(dyn, nd):
                    for b in combinations(stat, k - nd):
                        out.append(tuple(sorted(a + b)))
        return out

    def _allocate(self, n_tasks: int, caps: dict[str, int]) -> dict[str, int]:
        """三档大致均衡分配；纯档超上限的名额转入混合档。"""
        base = n_tasks // 3
        quota = {
            self.TAG_DYNAMIC: base,
            self.TAG_STATIC: base,
            self.TAG_MIXED: base + (n_tasks - 3 * base),  # 余数给混合档（如 16→5/5/6）
        }
        for tag in (self.TAG_DYNAMIC, self.TAG_STATIC):
            excess = quota[tag] - caps[tag]
            if excess > 0:
                quota[tag] = caps[tag]
                quota[self.TAG_MIXED] += excess
        if quota[self.TAG_MIXED] > caps[self.TAG_MIXED]:
            raise ValueError(
                f"n_tasks={n_tasks} 超过三档不重复类别组合总容量 "
                f"{sum(caps.values())}（纯动 {caps[self.TAG_DYNAMIC]} / "
                f"纯静 {caps[self.TAG_STATIC]} / 混合 {caps[self.TAG_MIXED]}）"
            )
        return quota

    def propose(
        self, y: np.ndarray, pool_indices: np.ndarray, n_tasks: int,
        rng: np.random.Generator,
    ) -> list[TaskSpec]:
        observed = set(np.unique(y[pool_indices]).tolist())
        declared = set(self.dynamic_classes) | set(self.static_classes)
        missing = declared - observed
        if missing:
            raise ValueError(
                f"池内缺少声明的类别 {sorted(missing)}——动/静分组是 HAR 语义，"
                "请确认数据集/标签或改用 RandomClassSubsetSlicer"
            )

        combos = {
            self.TAG_DYNAMIC: self._pure_combos(self.dynamic_classes),
            self.TAG_STATIC: self._pure_combos(self.static_classes),
            self.TAG_MIXED: self._mixed_combos(),
        }
        quota = self._allocate(n_tasks, {t: len(c) for t, c in combos.items()})

        specs: list[TaskSpec] = []
        for tag in (self.TAG_DYNAMIC, self.TAG_STATIC, self.TAG_MIXED):
            pool_combos = combos[tag]
            picked = rng.choice(len(pool_combos), size=quota[tag], replace=False)
            for i in picked:
                subset = pool_combos[int(i)]
                specs.append(TaskSpec(
                    classes=subset,
                    eligible=_eligible(y, pool_indices, subset),
                    tag=tag,
                ))
        return specs
