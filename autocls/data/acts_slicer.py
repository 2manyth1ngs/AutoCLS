"""AutoCTS++ 式任务划分（WP17-ACTS，results/acts/s1/PROTOCOL.md §1.1）。

对应 AutoCTS_plusplus/exps/dataset_slice.py L27–101：两维（变量子集 × 连续时间窗）各按
25 / 50 / 75 / 100% 四档分层、每格 ⌈n/16⌉ 个、变量不放回随机、窗起点随机、子集之间不去重。
分类语料的严格对应（方案 docs/AUTOCTS_STRICT_REPRO_PLAN.md §3.1）：

- 通道子集大小 k ∈ [channel_min, C] 四档；实例内连续时间窗长 w ∈ [window_min, L] 四档，
  起点 t0 ~ U[0, L − w]，**全部实例同一窗**（他们全部变量同一段时间）；
- 不切类别（他们没有标签集合轴）；样本取 min(n_samples_max, N) 个不放回随机；
- 档位候选 = 分位区间内的整数（前三档右开、末档闭）。**论文未给全，按 X 语义补全**：源码公式
  ``variable_s = l + round((i − gap)(r − l))`` 在 C 小时给空区间（C = 3 直接崩），这里改为
  "区间内无整数时取区间中点四舍五入"，C = 3 → {1, 2, 2, 3}，末档闭保证全通道 / 全长可达；
- 档位顺序 ``tier_order`` 让任意前缀在两维上尽量均衡（前 4 个每档各一次、前 8 个每档各两次），
  第 17 个起循环；**每任务独立 RNG 流**（``[seed, run_seed_for(task_id, "acts_slice")]``），
  改 n_tasks 不改任何已有任务（S1 试点 8 个 = S2 的 16 个的前缀）。

本模块只吃 y / C / L（铁律 7：不引入数据集概念）；旧的 ``slice_pool_tasks`` 路径一字不动。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..labeling import run_seed_for
from ..task_types import TASK_CLASSIFICATION, validate_task_type
from .task_slicer import SlicedTask

DEFAULT_TIER_ORDER: tuple[tuple[int, int], ...] = (
    (0, 0), (3, 3), (1, 2), (2, 1), (0, 3), (3, 0), (1, 1), (2, 2),
    (0, 1), (1, 0), (2, 3), (3, 2), (0, 2), (2, 0), (1, 3), (3, 1),
)
ID_PREFIX = "a"          # task_id = f"{pool}_a{i:02d}"（与注册表流 _NN、扩展流 _xNN 不撞）


def tier_candidates(lo: int, hi: int, tier: int, n_tiers: int = 4) -> list[int]:
    """档 tier 的整数候选：[lo, hi] 按 n_tiers 等分，前 n−1 档右开、末档闭；空区间取中点四舍五入。"""
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


@dataclass(frozen=True)
class AutoCTSTierSlicer:
    """通道档 × 窗长档 × 全类别的切片器（参数唯一真源 configs/acts.yaml: slicing）。"""

    seed: int = 0
    tiers: int = 4
    channel_min: int = 1
    window_min: int = 50
    n_samples_max: int = 1024
    min_samples: int = 32
    tier_order: tuple[tuple[int, int], ...] = DEFAULT_TIER_ORDER
    id_prefix: str = ID_PREFIX

    @classmethod
    def from_config(cls, sc: dict) -> "AutoCTSTierSlicer":
        order = sc.get("tier_order")
        return cls(
            seed=int(sc.get("seed", 0)), tiers=int(sc.get("tiers", 4)),
            channel_min=int(sc.get("channel_min", 1)), window_min=int(sc.get("window_min", 50)),
            n_samples_max=int(sc.get("n_samples_max", 1024)), min_samples=int(sc.get("min_samples", 32)),
            tier_order=tuple(tuple(int(x) for x in p) for p in order) if order else DEFAULT_TIER_ORDER,
        )

    def __post_init__(self) -> None:
        if not self.tier_order:
            raise ValueError("tier_order 不能为空")
        for ci, ti in self.tier_order:
            if not (0 <= ci < self.tiers and 0 <= ti < self.tiers):
                raise ValueError(f"tier_order 含越界档位 {(ci, ti)}（tiers={self.tiers}）")

    def slice(
        self, y: np.ndarray, *, n_channels: int, seq_len: int, n_tasks: int,
        pool_name: str, task_type: str = TASK_CLASSIFICATION,
    ) -> list[SlicedTask]:
        validate_task_type(task_type)
        y = np.asarray(y)
        n = int(len(y))
        if n < self.min_samples:
            raise ValueError(f"样本数 {n} < min_samples={self.min_samples}")
        if int(seq_len) < self.window_min:
            raise ValueError(f"seq_len={seq_len} < window_min={self.window_min}（encoder 适用边界）")
        if int(n_channels) < self.channel_min:
            raise ValueError(f"n_channels={n_channels} < channel_min={self.channel_min}")
        classes = tuple(int(v) for v in np.unique(y))
        out: list[SlicedTask] = []
        for i in range(int(n_tasks)):
            ci, ti = self.tier_order[i % len(self.tier_order)]
            tid = f"{pool_name}_{self.id_prefix}{i:02d}"
            rng = np.random.default_rng([int(self.seed), run_seed_for(tid, "acts_slice")])
            # 抽取顺序固定（协议 §1.1）：通道数 k → 通道集合 → 窗长 w → 起点 t0 → 样本
            k = int(rng.choice(tier_candidates(self.channel_min, int(n_channels), ci, self.tiers)))
            channels = tuple(sorted(int(c) for c in rng.choice(int(n_channels), size=k, replace=False)))
            w = int(rng.choice(tier_candidates(self.window_min, int(seq_len), ti, self.tiers)))
            t0 = int(rng.integers(0, int(seq_len) - w + 1))
            m = min(int(self.n_samples_max), n)
            samples = tuple(sorted(int(s) for s in rng.choice(n, size=m, replace=False)))
            out.append(SlicedTask(
                task_id=tid, pool=pool_name, classes=classes, channels=channels,
                sample_indices=samples, tag=f"c{ci}t{ti}", task_type=task_type,
                time_window=(t0, t0 + w),
            ))
        return out


__all__ = ["AutoCTSTierSlicer", "DEFAULT_TIER_ORDER", "ID_PREFIX", "tier_candidates"]
