"""AutoCTS++ 式任务划分（WP17-ACTS，results/acts/s1/PROTOCOL.md §1.1 / §1.9）：三轴四档 OA(16, 4³)。

对应 AutoCTS_plusplus/exps/dataset_slice.py L27–101：两维（变量子集 × 连续时间窗）各按 25 / 50 / 75 / 100%
四档分层、每格 ⌈n/16⌉ 个、变量不放回随机、窗起点随机、子集之间不去重。**映射更正（用户 2026-09-19）**：他们的
连续时间窗轴在预测语料里的语义是训练样本量，对应本方案的样本数 n；实例内时间窗 L 是分类特有的附加轴。三轴：

- 通道子集大小 k ∈ [channel_min, C]，线性四档；
- 实例内连续时间窗长 w ∈ [window_min, L]，线性四档，起点 t0 ~ U[0, L − w]，**全部实例同一窗**；
- 样本数 n ∈ [max(n_min, n_min_per_class·k), min(n_cap, N)]，**对数四档**，按类分层抽样（每类先 n_min_per_class 个，
  余量按类频率比例、最大余数法分配，类内不放回随机、类按 id 升序逐类抽）；
- 不切类别（他们没有标签集合轴）。

档位候选 = 分位区间内的整数（前三档右开、末档闭）。**论文未给全，按 X 语义补全**：源码公式
``variable_s = l + round((i − gap)(r − l))`` 在 C 小时给空区间（C = 3 直接崩），这里改为"区间内无整数时取
区间中点（对数档取几何中点）四舍五入"，C = 3 → {1, 2, 2, 3}，末档闭保证全通道 / 全长 / 上限样本数可达。

三轴设计 = OA(16, 4³)（强度 2，任两列的 16 个档位组合各出现一次）：GF(4) 加法（整数表示下 = 按位异或）构造，
行 (a, b, a ⊕ b) = (通道档, 窗长档, 样本档)，行序按 t = 0..3 分块、块内 b = α·a ⊕ t（α 为生成元：0→0, 1→2,
2→3, 3→1），使前 4 / 8 / 12 行在三轴上各档等频；试点 8 = 前 8 行；第 17 个起循环。**每任务独立 RNG 流**
（``[seed, run_seed_for(task_id, "acts_slice")]``），改 n_tasks 不改任何已有任务。

本模块只吃 y / C / L（铁律 7：不引入数据集概念）；旧的 ``slice_pool_tasks`` 路径一字不动。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..labeling import run_seed_for
from ..task_types import TASK_CLASSIFICATION, validate_task_type
from .task_slicer import SlicedTask

ID_PREFIX = "a"          # task_id = f"{pool}_a{i:02d}"（与注册表流 _NN、扩展流 _xNN 不撞）
_GF4_ALPHA = {0: 0, 1: 2, 2: 3, 3: 1}      # GF(4) 里乘生成元 α（元素 {0, 1, α, α²} ↔ {0, 1, 2, 3}）


def oa16_design() -> tuple[tuple[int, int, int], ...]:
    """OA(16, 4³) 强度 2：行 (a, b, a ⊕ b)，b = α·a ⊕ t，块 t = 0..3 → 前 4 / 8 / 12 行各轴等频。"""
    rows: list[tuple[int, int, int]] = []
    for t in range(4):
        for a in range(4):
            b = _GF4_ALPHA[a] ^ t
            rows.append((a, b, a ^ b))
    return tuple(rows)


DEFAULT_DESIGN: tuple[tuple[int, int, int], ...] = oa16_design()


def is_orthogonal_array(rows: Sequence[Sequence[int]], levels: int = 4) -> bool:
    """强度 2 正交表判定：行数 = levels²，任两列的 (档, 档) 组合各出现一次。"""
    rows = [tuple(int(v) for v in r) for r in rows]
    if len(rows) != levels ** 2 or not rows:
        return False
    k = len(rows[0])
    if any(len(r) != k for r in rows):
        return False
    for i in range(k):
        for j in range(i + 1, k):
            if len({(r[i], r[j]) for r in rows}) != levels ** 2:
                return False
    return True


def tier_candidates(lo: int, hi: int, tier: int, n_tiers: int = 4) -> list[int]:
    """线性档：[lo, hi] 按 n_tiers 等分，前 n−1 档右开、末档闭；空区间取中点四舍五入。"""
    lo, hi, tier, n_tiers = int(lo), int(hi), int(tier), int(n_tiers)
    if hi < lo:
        raise ValueError(f"档位区间上界 {hi} < 下界 {lo}")
    if not 0 <= tier < n_tiers:
        raise ValueError(f"tier={tier} 须在 [0, {n_tiers})")
    span = hi - lo
    a = lo + span * tier / n_tiers
    b = lo + span * (tier + 1) / n_tiers
    last = tier == n_tiers - 1
    cand = [k for k in range(lo, hi + 1) if (a <= k <= b if last else a <= k < b)]
    return cand or [int(round((a + b) / 2))]


def log_tier_candidates(lo: int, hi: int, tier: int, n_tiers: int = 4) -> list[int]:
    """对数档：[lo, hi] 按比例 r = hi/lo 几何等分，档 i 覆盖 [lo·r^{i/n}, lo·r^{(i+1)/n}]（前 n−1 档右开、末档闭）；
    空区间取几何中点四舍五入。lo == hi 时四档都是 [lo]。"""
    lo, hi, tier, n_tiers = int(lo), int(hi), int(tier), int(n_tiers)
    if hi < lo or lo <= 0:
        raise ValueError(f"对数档位区间须 0 < lo ≤ hi，得到 [{lo}, {hi}]")
    if not 0 <= tier < n_tiers:
        raise ValueError(f"tier={tier} 须在 [0, {n_tiers})")
    r = hi / lo
    a = float(lo) if tier == 0 else lo * r ** (tier / n_tiers)
    last = tier == n_tiers - 1
    b = float(hi) if last else lo * r ** ((tier + 1) / n_tiers)
    cand = [k for k in range(lo, hi + 1) if (a <= k <= b if last else a <= k < b)]
    return cand or [int(round(math.sqrt(a * b)))]


def stratified_sample(y: np.ndarray, *, n: int, rng: np.random.Generator, min_per_class: int = 2) -> tuple[int, ...]:
    """按类分层抽 n 个：目标 = n·(类频率)，取整后每类抬到 ≥ min_per_class（不超可用数），差额按最大余数补 / 按最大
    超出量减，类按 id 升序逐类不放回随机。返回升序全局索引。"""
    y = np.asarray(y)
    classes = [int(c) for c in np.unique(y)]
    k, N, n = len(classes), int(len(y)), int(n)
    counts = {c: int((y == c).sum()) for c in classes}
    if n > N:
        raise ValueError(f"n={n} > 样本数 {N}")
    if n < k * int(min_per_class):
        raise ValueError(f"n={n} < 类数 {k} × min_per_class={min_per_class}")
    short = [c for c in classes if counts[c] < int(min_per_class)]
    if short:
        raise ValueError(f"类 {short} 的样本数 < min_per_class={min_per_class}")
    target = {c: n * counts[c] / N for c in classes}
    alloc = {c: max(int(min_per_class), min(counts[c], int(math.floor(target[c])))) for c in classes}
    frac = {c: target[c] - math.floor(target[c]) for c in classes}
    diff = n - sum(alloc.values())
    guard = 0
    while diff != 0:
        guard += 1
        if guard > 4 * n + 4 * k:
            raise RuntimeError("分层分配未收敛")
        if diff > 0:
            cands = [c for c in classes if alloc[c] < counts[c]]
            c = max(cands, key=lambda c: (frac[c], -c))
            alloc[c] += 1
            diff -= 1
        else:
            cands = [c for c in classes if alloc[c] > int(min_per_class)]
            c = max(cands, key=lambda c: (alloc[c] - target[c], -c))
            alloc[c] -= 1
            diff += 1
    idx: list[int] = []
    for c in classes:
        pool = np.where(y == c)[0]
        idx.extend(int(v) for v in rng.choice(pool, size=alloc[c], replace=False))
    return tuple(sorted(idx))


@dataclass(frozen=True)
class AutoCTSTierSlicer:
    """通道档 × 窗长档 × 样本数档（OA(16, 4³)）× 全类别的切片器（参数唯一真源 configs/acts.yaml: slicing）。"""

    seed: int = 0
    tiers: int = 4
    channel_min: int = 1
    window_min: int = 50
    n_min: int = 32
    n_min_per_class: int = 2
    n_cap: Optional[int] = 1024
    min_samples: int = 32
    design: tuple[tuple[int, int, int], ...] = DEFAULT_DESIGN
    id_prefix: str = ID_PREFIX

    @classmethod
    def from_config(cls, sc: dict) -> "AutoCTSTierSlicer":
        design = sc.get("design")
        cap = sc.get("n_cap", 1024)
        return cls(
            seed=int(sc.get("seed", 0)), tiers=int(sc.get("tiers", 4)),
            channel_min=int(sc.get("channel_min", 1)), window_min=int(sc.get("window_min", 50)),
            n_min=int(sc.get("n_min", 32)), n_min_per_class=int(sc.get("n_min_per_class", 2)),
            n_cap=(None if cap is None else int(cap)), min_samples=int(sc.get("min_samples", 32)),
            design=tuple(tuple(int(x) for x in row) for row in design) if design else DEFAULT_DESIGN,
        )

    def __post_init__(self) -> None:
        if not self.design:
            raise ValueError("design 不能为空")
        for row in self.design:
            if len(row) != 3 or not all(0 <= v < self.tiers for v in row):
                raise ValueError(f"design 行 {row} 须是三轴档位且各在 [0, {self.tiers})")

    def sample_bounds(self, n_samples: int, n_classes: int) -> tuple[int, int]:
        """样本数轴区间 [max(n_min, n_min_per_class·k), min(n_cap, N)]。"""
        n_lo = max(int(self.n_min), int(self.n_min_per_class) * int(n_classes))
        n_hi = int(n_samples) if self.n_cap is None else min(int(self.n_cap), int(n_samples))
        return n_lo, n_hi

    def slice(
        self, y: np.ndarray, *, n_channels: int, seq_len: int, n_tasks: int,
        pool_name: str, task_type: str = TASK_CLASSIFICATION,
    ) -> list[SlicedTask]:
        validate_task_type(task_type)
        y = np.asarray(y)
        n = int(len(y))
        classes = tuple(int(v) for v in np.unique(y))
        n_lo, n_hi = self.sample_bounds(n, len(classes))
        if n < self.min_samples or n_hi < n_lo:
            raise ValueError(f"样本数 N={n}：样本数轴 n_lo={n_lo} > n_hi={n_hi}（min_samples={self.min_samples}）")
        if int(seq_len) < self.window_min:
            raise ValueError(f"seq_len={seq_len} < window_min={self.window_min}（encoder 适用边界）")
        if int(n_channels) < self.channel_min:
            raise ValueError(f"n_channels={n_channels} < channel_min={self.channel_min}")
        out: list[SlicedTask] = []
        for i in range(int(n_tasks)):
            ci, wi, ni = self.design[i % len(self.design)]
            tid = f"{pool_name}_{self.id_prefix}{i:02d}"
            rng = np.random.default_rng([int(self.seed), run_seed_for(tid, "acts_slice")])
            # 抽取顺序固定（协议 §1.1）：通道数 → 通道集合 → 窗长 w → 起点 t0 → 样本数 n → 按类分层抽样
            k_ch = int(rng.choice(tier_candidates(self.channel_min, int(n_channels), ci, self.tiers)))
            channels = tuple(sorted(int(c) for c in rng.choice(int(n_channels), size=k_ch, replace=False)))
            w = int(rng.choice(tier_candidates(self.window_min, int(seq_len), wi, self.tiers)))
            t0 = int(rng.integers(0, int(seq_len) - w + 1))
            m = int(rng.choice(log_tier_candidates(n_lo, n_hi, ni, self.tiers)))
            samples = stratified_sample(y, n=m, rng=rng, min_per_class=self.n_min_per_class)
            out.append(SlicedTask(
                task_id=tid, pool=pool_name, classes=classes, channels=channels,
                sample_indices=samples, tag=f"c{ci}w{wi}n{ni}", task_type=task_type,
                time_window=(t0, t0 + w),
            ))
        return out


__all__ = ["AutoCTSTierSlicer", "DEFAULT_DESIGN", "ID_PREFIX", "is_orthogonal_array", "log_tier_candidates",
           "oa16_design", "stratified_sample", "tier_candidates"]
