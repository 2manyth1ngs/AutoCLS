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


def wilson_interval(k: int, n: int, z: float = 1.645) -> tuple[float, float]:
    """比例 k/n 的 Wilson score 区间（默认 z=1.645 → 90%，与既有报告口径一致）。

    workpoint followup 协议 §1 的确认门槛由它导出：n=60 时下界脱离 0.5 需
    k≥37（37→0.511，36→0.494）。REPORT_STEP3 §2 与 iters_workpoint REPORT §2.1
    的引用值（12/20→[0.42,0.76]、15/20→[0.57,0.87]）均可由本函数复算。
    """
    if n <= 0 or k < 0 or k > n:
        raise ValueError(f"非法计数 k={k}, n={n}")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def confirm_verdict(
    *, n_above: int, n_below: int, n_trained: int, t1: bool, t2: bool,
    z: float = 1.645,
) -> dict:
    """加测确认层级（followup 协议 §1，n=60 口径；用户拍板 B 层不算入池）。

    - A confirmed：T1+T2+T3 全过 **且** ↑/n 的 Wilson 90% 下界 >0.5 → 可作入池依据；
    - B marginal：T1+T2+T3 过但 Wilson 仍含 0.5 → usable 判定维持、统计边缘未消除，
      **不作入池依据**（用户裁决，2026-08-10）；
    - C reversed：T1/T2/T3 任一不过 → 更大样本优先，20 条的 usable 判定撤销。
    T3 阈值直接引用判据 v2 常数（同一实现，禁止两处漂移）。
    """
    from autocls.baseline_probe import T3_ABOVE_FRACTION, T3_ABOVE_OVER_BELOW

    t3 = bool(n_trained > 0 and n_above >= T3_ABOVE_FRACTION * n_trained
              and n_above >= T3_ABOVE_OVER_BELOW * n_below)
    lo = hi = None
    if n_trained > 0:
        lo, hi = wilson_interval(n_above, n_trained, z)
    if not (t1 and t2 and t3):
        tier = "reversed"
    elif lo is not None and lo > 0.5:
        tier = "confirmed"
    else:
        tier = "marginal"
    return {"tier": tier, "t3": t3, "wilson_low": lo, "wilson_high": hi,
            "pool_eligible": tier == "confirmed"}


def select_workpoint(arms: dict[int, dict], *, margin: float = 0.02) -> Optional[dict]:
    """确认工作点选择（workpoint_confirm 协议 §2；用户裁决"0.02 让分窗"，2026-08-11）。

    输入 {iters: {"usable": T1+T2+T3 全过, "best_gain": float}}——扫描臂与历史臂
    （如 UWave@290）平等入表。数据前写死的规则：
    1. 候选 = usable 且 best_gain 有限的臂；空 → None（无可确认点）。
    2. 取 best_gain 最大者；与最大者差 ≤ margin 的候选视为并列，并列取最少 iters。
       让分只对"与最大者的差"生效，不做传递闭包。
    理由（预注册）：历史臂的 best_gain 带跨硬件 ±1 样本抖动（rprime 5090 记忆条目），
    幅度差在一两个量子内不可信；破坏参与随 iters 单调涨（followup §4 规律 2），
    同幅度下更早的臂 label 成分更好——Handwriting 弃 290 选 20–40ep 的先例同构。
    """
    usable = {
        it: float(a["best_gain"]) for it, a in arms.items()
        if a.get("usable") and a.get("best_gain") is not None
        and math.isfinite(a["best_gain"])
    }
    if not usable:
        return None
    top = max(usable.values())
    tied = [it for it, g in usable.items() if top - g <= margin]
    pick = min(tied)
    return {"iters": pick, "best_gain": usable[pick],
            "top_gain": top, "candidates": sorted(usable)}


def destruction_participation(
    per_task: dict[str, dict[str, Optional[float]]],
    baselines: dict[str, tuple[float, float]],
    *, delta: float,
) -> dict:
    """δ 过滤后存活配对里"破坏参与"的占比（followup 协议 §1；如实报告，不设硬门槛）。

    per_task: {task_id: {cls_key: R'}}（None/NaN = 失败记录，剔除）；
    baselines: {task_id: (基线均值, 带半宽)}——带半宽必须喂 summarize_task 的
    null_band（2·max(σ, 1/n_eval)），保证"strict 破坏"与 ↑/↓ 计数是同一条带子。
    配对只在任务内（造 label 的口径），存活 = |ΔR'| ≥ δ。
    - strict：至少一方 R' < 均值 − 带（显著破坏，解耦实验的 94.7–100% 用的口径）；
    - loose：至少一方 R' < 均值。
    """
    n_pairs = n_survive = n_strict = n_loose = 0
    n_cls = n_below_band = n_below_mean = 0
    for tid, cls_map in per_task.items():
        mean, band = baselines[tid]
        vals = {k: float(v) for k, v in cls_map.items()
                if v is not None and v == v}
        n_cls += len(vals)
        strict = {k for k, v in vals.items() if v < mean - band}
        loose = {k for k, v in vals.items() if v < mean}
        n_below_band += len(strict)
        n_below_mean += len(loose)
        keys = sorted(vals)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                n_pairs += 1
                if abs(vals[a] - vals[b]) < delta:
                    continue
                n_survive += 1
                n_strict += a in strict or b in strict
                n_loose += a in loose or b in loose
    return {
        "n_cls": n_cls, "n_below_band": n_below_band, "n_below_mean": n_below_mean,
        "n_pairs": n_pairs, "n_survive": n_survive,
        "survive_rate": (n_survive / n_pairs) if n_pairs else None,
        "n_strict": n_strict,
        "strict_share": (n_strict / n_survive) if n_survive else None,
        "n_loose": n_loose,
        "loose_share": (n_loose / n_survive) if n_survive else None,
    }


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
