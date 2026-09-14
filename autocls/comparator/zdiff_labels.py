"""z 差分软标签（WP16-TCLSC-L3，协议 results/tclsc_l3/PROTOCOL.md §2）。

δ 配对硬标签只保留 |Δr| ≥ δ 的对并把方向压成 0/1（C2：存活对任务盲天花板 0.95+，任务信号被滤掉）。
本模块给出另一种标签：任务内**全部**无序对、不过 δ、不弃平局，目标

    p = σ(k · d)，  d = |z_a − z_b| × sign，  sign = +1 若 better_fn(r_a, r_b)，−1 若 better_fn(r_b, r_a)，否则 0，

z 是任务内（engine task = 切片 × 预算）R' 的 z 分数（同一任务内标准化，尺度不变）。方向只经 `better_fn`
判定（铁律 8），本模块不知道任务类型、数据集或"高好还是低好"（铁律 7）。样本权 w = 1；无序对的呈现方向由 rng
逐 epoch 抛硬币（concat 头需要，反对称头无关）。输出 `soft_labels.SoftPair`，走 `SoftTCLSCEngine.train_soft`。
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Callable, Optional, Sequence

import numpy as np

from .soft_labels import SoftPair
from .tclsc_engine import ClsRecord

BetterFn = Callable[[float, float], bool]


def within_task_z(records: Sequence[ClsRecord]) -> dict[tuple[str, str], float]:
    """{(task_id, key): z}——每个任务内按 r 标准化；任务内 sd=0 时全 0。"""
    by_task: dict[str, list[ClsRecord]] = {}
    for r in records:
        by_task.setdefault(r.task_id, []).append(r)
    out: dict[tuple[str, str], float] = {}
    for tid, rs in by_task.items():
        v = np.array([float(r.r) for r in rs])
        mu, sd = float(v.mean()), float(v.std())
        for r in rs:
            out[(tid, r.key)] = (float(r.r) - mu) / sd if sd > 0 else 0.0
    return out


def zdiff_pairs(
    records: Sequence[ClsRecord],
    better_fn: BetterFn,
    *,
    k: float,
    rng: np.random.Generator,
    origins: Optional[tuple[str, ...]] = None,
) -> list[SoftPair]:
    """任务内全部无序对 → SoftPair（p = σ(k·d)，w = 1，呈现方向随机）。origins 给出时只用该来源的记录。"""
    if k <= 0:
        raise ValueError(f"k 须 >0，得到 {k}")
    if origins is not None:
        records = [r for r in records if r.origin in origins]
    z = within_task_z(records)
    by_task: dict[str, list[ClsRecord]] = {}
    for r in records:
        by_task.setdefault(r.task_id, []).append(r)
    out: list[SoftPair] = []
    for tid in sorted(by_task):
        rs = sorted(by_task[tid], key=lambda r: r.key)
        for a, b in combinations(rs, 2):
            if a.key == b.key:
                continue
            if rng.random() < 0.5:
                a, b = b, a
            if better_fn(a.r, b.r):
                sign = 1.0
            elif better_fn(b.r, a.r):
                sign = -1.0
            else:
                sign = 0.0
            d = abs(z[(tid, a.key)] - z[(tid, b.key)]) * sign
            p = 1.0 / (1.0 + math.exp(-float(k) * d))
            out.append(SoftPair(task_id=tid, key_a=a.key, key_b=b.key, vec_a=a.vec, vec_b=b.vec,
                                p=float(p), w=1.0, y_hard=int(sign > 0),
                                kind="-".join(sorted((a.origin, b.origin)))))
    return out
