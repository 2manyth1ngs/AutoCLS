"""iters 工作点扫描的判定逻辑（results/iters_workpoint/PROTOCOL.md §4）。

纯函数：吃 arm 摘要（由 autocls.baseline_probe.summarize_* 产出），不碰缓存与
数据集。阈值全部来自协议锁，跑完不调；驱动脚本 scripts/iters_workpoint_scan.py
只负责喂数据与打表。

口径要点（协议 §0.1）：
- steps/epoch = ⌈n_train / batch_size⌉，bs=128（train.yaml）。数据集内 steps/epoch
  恒定（候选池任务一律 716–718 → 6，Handwriting 前 2 任务 → 3），故 epoch 对齐
  退化为 per-dataset iters。
- "最优工作点"按 **best_gain** 的 argmax 定义（判据 2，用户拍板）：下游造 label
  需要的是可分辨幅度（best_gain ≥ δ 才有存活配对）；realized gain 只作辅助诊断。
"""
from __future__ import annotations

import math
from typing import Optional

EPOCH_BATCH_SIZE = 128  # = configs/train.yaml batch_size；铁律 2 的 iters 计量基准

# 判据 3 阈值：峰值 epoch 的 max/min 比。网格是 ~2× 对数间隔，更细分辨率不许声称。
PEAK_RATIO_CONSISTENT = 2.0
PEAK_RATIO_INCONSISTENT = 4.0

# 判据 4：翻盘（稳）的单点余量。T3 门槛是 ≥10/20，恰 10 时 Wilson 90% 含 0.5
# 且 4–5 点多重比较放大假阳性 → 单点须 ≥11 才算稳。
FLIP_SOLID_N_ABOVE = 11


def steps_per_epoch(n_train: int, batch_size: int = EPOCH_BATCH_SIZE) -> int:
    """一个 epoch 的参数更新步数。cl_trainer 按 ⌈n/bs⌉ 个 batch 遍历一轮。"""
    if n_train <= 0:
        raise ValueError(f"n_train 必须为正，得到 {n_train}")
    return math.ceil(n_train / batch_size)


def arm_epoch(iters: int, spe: int) -> float:
    """iters 臂在该任务口径下的等效 epoch 数。"""
    return iters / spe


def find_peak(arms: dict[int, Optional[float]]) -> Optional[dict]:
    """best_gain 的 argmax（判据 2）。

    - None 值（未测/测不满的臂）不参与；全 None 返回 None。
    - 平局取更小 iters：更便宜的工作点，且保证确定性。
    - 峰落在扫描网格端点 → censored "left"/"right"（真峰可能在网格外，报告层
      只许报 "≥/≤ 端点"）；只有一个点则 "both"。
    """
    valid = {it: v for it, v in arms.items() if v is not None and math.isfinite(v)}
    if not valid:
        return None
    grid = sorted(valid)
    peak_iters = min(grid, key=lambda it: (-valid[it], it))
    if len(grid) == 1:
        censored = "both"
    elif peak_iters == grid[0]:
        censored = "left"
    elif peak_iters == grid[-1]:
        censored = "right"
    else:
        censored = None
    return {"iters": peak_iters, "value": valid[peak_iters], "censored": censored}


def peak_consistency(peaks: dict[str, tuple[float, Optional[str]]]) -> dict:
    """跨数据集峰值 epoch 一致性（判据 3）。

    输入 {数据集: (峰值 epoch, censored)}，只应包含"至少一个臂 best_gain ≥ δ"的
    数据集（过滤在调用方——平坦噪声曲线的 argmax 无意义）。
    协议原文：≤2 → 一致；≥4 → 不一致；之间**或有截尾峰参与** → 存疑。
    """
    if len(peaks) < 2:
        return {"verdict": "insufficient", "ratio": None, "n": len(peaks)}
    eps = [e for e, _c in peaks.values()]
    ratio = max(eps) / min(eps)
    if any(c is not None for _e, c in peaks.values()):
        verdict = "uncertain"
    elif ratio <= PEAK_RATIO_CONSISTENT:
        verdict = "consistent"
    elif ratio >= PEAK_RATIO_INCONSISTENT:
        verdict = "inconsistent"
    else:
        verdict = "uncertain"
    return {"verdict": verdict, "ratio": ratio, "n": len(peaks)}


def classify_flip(arms: dict[int, dict]) -> dict:
    """候选翻盘判定（判据 4）。

    输入 {iters: {"usable": T1+T2+T3 全过, "n_above": ↑ 条数}}。
    - 稳：任一 usable 点 ↑ ≥ 11，或**相邻**两个网格点都 usable
      （两点独立撞线概率平方级，相邻性防"多重比较捞出的孤点"）；
    - 弱：仅有单点（或非相邻多点）恰 10/20 —— 只算"需加测确认"，不作入池依据。
    """
    grid = sorted(arms)
    usable = [it for it in grid if arms[it].get("usable")]
    if not usable:
        return {"flip": "none", "at": []}
    if any(arms[it].get("n_above", 0) >= FLIP_SOLID_N_ABOVE for it in usable):
        return {"flip": "solid", "at": usable}
    adjacent = any(
        grid[i + 1] in usable
        for i, it in enumerate(grid[:-1]) if it in usable
    )
    return {"flip": "solid" if adjacent else "weak", "at": usable}


def no_workpoint_verdict(arms: dict[int, dict], *, quantum: float) -> dict:
    """"无工作点"双维确认（判据 5，用户拍板补 best_gain 维度）。

    输入 {iters: {"realized_gain": float, "T2": bool, "T3": bool}}。
    - (a) 净收益维空：所有臂 realized gain ≤ 0 + quantum（一个 ACC 量子 1/n_eval，
      数不出一个样本的差不算收益；NaN 不算收益）；
    - (b) label 产量维空：没有任何臂同时过 T2+T3。
    a、b 都空 → no_workpoint；a 空 b 不空 → label_workpoint_only（造 label 可用，
    与"无工作点"必须分开登记）；a 不空 → has_net_gain。
    """
    def _gain_ok(v) -> bool:
        return v is None or not math.isfinite(v) or v <= quantum

    net_empty = all(_gain_ok(a.get("realized_gain")) for a in arms.values())
    label_empty = not any(a.get("T2") and a.get("T3") for a in arms.values())
    if not net_empty:
        verdict = "has_net_gain"
    elif label_empty:
        verdict = "no_workpoint"
    else:
        verdict = "label_workpoint_only"
    return {"verdict": verdict, "net_gain_dim_empty": net_empty,
            "label_dim_empty": label_empty}
