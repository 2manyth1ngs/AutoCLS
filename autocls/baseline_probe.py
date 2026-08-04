"""随机初始化基线探针的纯逻辑（协议 `results/baseline_probe/PROTOCOL.md`）。

**为什么要这个量**：解耦实验（results/decouple_random_cls/REPORT.md）证明
AWR/CT/ERing/NATOPS 的随机初始化基线贴顶（0.90–0.99），常规 CLS 的训练增益只有
±0.013 → R'（下游 ACC）测不出学习质量差异，只能测破坏程度。于是把"哪些数据集能用"
从统计判定（SNR 闸 / gate v2，两者都对探针 CLS 构成敏感）换成有机制解释的客观量：

    基线离顶的距离（headroom = 1 − baseline）= 学习质量的可测空间**上界**。

上界不等于可用空间：2 类 EEG 数据集 baseline≈0.51≈随机猜测，headroom 名义 0.49，
但若 290 iters 训练后 R̄ 仍是 0.51，这 0.49 一分也没被用上。故判据是**四象限**
（headroom × realized gain），不是单看 headroom（classify_quadrant）。

iters=0 的语义：`pretrain_encoder` 的 `while step < max_iters` 不进循环 → encoder
停在随机初始化，**CLS 完全不参与计算**。基线因此是 (任务, 划分, 初始化) 的属性，
与策略无关；`inert_cls()` 只是 evaluate_rprime 的必填占位（测试断言其无影响）。
"""
from __future__ import annotations

import statistics
from typing import Iterable, Optional, Sequence

from .data.source_pool import ROLE_RESERVE, ROLE_SOURCE, SourcePoolRegistry
from .labeling import run_seed_for
from .space.cls_space import CLS

__all__ = [
    "BASELINE_ITERS", "BEST_GAIN_FLOOR", "CEILING_BASELINE", "OPEN_BASELINE",
    "baseline_seed_specs", "classify_quadrant", "classify_tier", "inert_cls",
    "probe_scope", "summarize_dataset", "summarize_task",
]

# 训练量：0 = 不训练（铁律 2 要求 max_iters 显式传参；这里显式为 0）
BASELINE_ITERS = 0

# PROTOCOL.md §3.1 写死的三档阈值。典型 δ=0.05（labels.yaml: delta_per_dataset
# 跨数据集 0.03–0.08）：0.90 = headroom 2δ，0.70 = headroom 6δ。
CEILING_BASELINE = 0.90
OPEN_BASELINE = 0.70
# §3.3：best gain 门槛取典型 δ——增益低于 δ 的配对产不出存活 label。
BEST_GAIN_FLOOR = 0.05


def baseline_seed_specs(task_id: str, s: int) -> tuple[int, int]:
    """第 s 个基线运行的 (weight_seed, split_seed)。

    split **恒为阶段1 canonical 划分**：基线要当 canonical R' 的零训练对照相减，
    两者必须落在同一 train/eval 切分上，否则差值里混进划分方差。
    weight 每个 s 派生一个 → S 个独立随机初始化，报均值±std。
    """
    return run_seed_for(task_id, f"baseline:{s}"), run_seed_for(task_id, "train_eval_split")


def inert_cls() -> CLS:
    """evaluate_rprime 的必填占位 CLS。iters=0 下增强/损失都在未进入的循环体内，
    它对结果零影响（tests/test_baseline_probe.py 断言两条极端 CLS 逐位同分）。
    取 space.yaml 合法的最小路径策略，保证能被序列化进缓存。"""
    return CLS(
        resize=0.0, rescale=0.0, jitter=0.3, point_mask=0.0, freq_mask=0.0,
        crop=0.5, order=0,
        emb_jitter=0.0, emb_mask=0.0, norm="none",
        instance=True, temporal=False, cross_scale=False,
        kernel=None, pool=None, adj_neighbor=None,
        loss_type="info_nce", sim_func="cosine", temperature=0.1,
    )


def probe_scope(registry: SourcePoolRegistry) -> list[str]:
    """探针范围 = 源池 ∪ reserve。**目标池绝不入内**（铁律 6）——本地有数据也不碰，
    保持 zero-shot 边界；调用方另有 load_source_dataset 的 fail-fast 兜底。"""
    return [e.name for e in registry.by_role(ROLE_SOURCE)] + \
           [e.name for e in registry.by_role(ROLE_RESERVE)]


def classify_tier(baseline: float) -> str:
    """三档：贴顶 / 中等 / 离顶（PROTOCOL.md §3.1）。"""
    if baseline > CEILING_BASELINE:
        return "ceiling"
    if baseline >= OPEN_BASELINE:
        return "mid"
    return "open"


def classify_quadrant(baseline: float, best_gain: Optional[float]) -> str:
    """四象限判据（PROTOCOL.md §3.3）：headroom 是上界，realized gain 才是可用空间。

    - ceiling：headroom<0.10，只能产破坏排序 label（gain 大小无关，故无需配对数据）；
    - usable：headroom 够 且 最好的 CLS 能拉开 ≥δ；
    - inert：headroom 够但没 CLS 用得上（伪离顶——低分不等于有信号）；
    - unknown：headroom 够但缺阶段1 配对 R'，不臆断。
    """
    if classify_tier(baseline) == "ceiling":
        return "ceiling"
    if best_gain is None:
        return "unknown"
    return "usable" if best_gain >= BEST_GAIN_FLOOR else "inert"


def _finite(values: Iterable[Optional[float]]) -> list[float]:
    """剔除 None（失败记录）与 NaN。失败是"这条没跑出来"的结论，不是 0 分。"""
    return [float(v) for v in values if v is not None and v == v]


def summarize_task(
    rows: Sequence[dict], *, n_classes: int, trained: Sequence[Optional[float]] = (),
) -> dict:
    """单任务汇总：S 个随机初始化的基线分布 + 与阶段1 canonical R' 的配对增益。

    rows: 该任务的基线缓存记录（含 r_prime，可能为 None）。
    trained: 该任务阶段1 canonical R'（290 iters，同 task_id/同 split）列表，可为空。
    """
    vals = _finite(r["r_prime"] for r in rows)
    tr = _finite(trained)
    mean = statistics.fmean(vals) if vals else float("nan")
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    # 零假设带宽：随机初始化自身的 ±2σ。best_gain 是"N 条 CLS 取最大"、天然向上偏，
    # 只有落在这条带子外的 CLS 才算真 beat/break 了随机表征。σ 用 ACC 量子 1/n_eval
    # 兜底——有的任务 5 个种子同分（σ=0），否则"超出 2σ"退化成"超出均值"。
    quantum = max((1.0 / r["n_eval"] for r in rows if r.get("n_eval")), default=0.0)
    band = 2.0 * max(std, quantum)
    out = {
        "n_seeds": len(vals),
        "n_failed": len(rows) - len(vals),
        "n_classes": int(n_classes),
        "baseline_mean": mean,
        "baseline_std": std,
        "baseline_min": min(vals) if vals else float("nan"),
        "baseline_max": max(vals) if vals else float("nan"),
        "headroom": 1.0 - mean,
        "chance": 1.0 / n_classes,
        "lift_over_chance": mean - 1.0 / n_classes,
        "tier": classify_tier(mean) if vals else "unknown",
        "n_trained": len(tr),
        "trained_mean": statistics.fmean(tr) if tr else None,
        "trained_max": max(tr) if tr else None,
        "realized_gain": (statistics.fmean(tr) - mean) if (tr and vals) else None,
        "best_gain": (max(tr) - mean) if (tr and vals) else None,
        "null_band": band,
        "n_trained_above_null": sum(1 for v in tr if v > mean + band) if vals else 0,
        "n_trained_below_null": sum(1 for v in tr if v < mean - band) if vals else 0,
    }
    out["quadrant"] = classify_quadrant(mean, out["best_gain"]) if vals else "unknown"
    return out


def summarize_dataset(tasks: dict[str, dict]) -> dict:
    """数据集汇总：任务级基线的分布（min/median/max）+ 数据集档与象限。

    数据集档按 **per-task 均值的均值**（任务等权，不按样本数加权——每个任务都是一条
    独立的 label 生产线）；象限用 best_gain 的**跨任务最大值**（"这个数据集上最好的
    (任务, CLS) 组合能拉多少"），宽松侧取值，避免把可救的数据集误判成 inert。
    """
    per_task = [t["baseline_mean"] for t in tasks.values()]
    gains = [t["best_gain"] for t in tasks.values() if t.get("best_gain") is not None]
    mean = statistics.fmean(per_task)
    tier_counts: dict[str, int] = {}
    for t in tasks.values():
        k = classify_tier(t["baseline_mean"])
        tier_counts[k] = tier_counts.get(k, 0) + 1
    best_gain_max = max(gains) if gains else None
    return {
        "n_tasks": len(tasks),
        "baseline_mean": mean,
        "baseline_spread": statistics.stdev(per_task) if len(per_task) > 1 else 0.0,
        "task_min": min(per_task),
        "task_median": statistics.median(per_task),
        "task_max": max(per_task),
        "headroom": 1.0 - mean,
        "chance_mean": statistics.fmean([t["chance"] for t in tasks.values()]),
        "lift_over_chance": mean - statistics.fmean([t["chance"] for t in tasks.values()]),
        "best_gain_max": best_gain_max,
        "best_gain_median": statistics.median(gains) if gains else None,
        # 零假设诊断（不参与判定，用于标注哪些 usable 判得脆）：跨任务累计有多少条
        # 训练后的 CLS 真的跑出了随机初始化的 ±2σ 带子。
        "n_trained_above_null": sum(t.get("n_trained_above_null", 0) for t in tasks.values()),
        "n_trained_below_null": sum(t.get("n_trained_below_null", 0) for t in tasks.values()),
        "n_trained_total": sum(t.get("n_trained", 0) for t in tasks.values()),
        "realized_gain_mean": statistics.fmean(
            [t["realized_gain"] for t in tasks.values() if t.get("realized_gain") is not None]
        ) if any(t.get("realized_gain") is not None for t in tasks.values()) else None,
        "tier": classify_tier(mean),
        "tier_counts": tier_counts,
        "quadrant": classify_quadrant(mean, best_gain_max),
    }
