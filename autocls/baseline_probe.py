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
    "MIN_N_CLASSES", "MIN_N_EVAL", "MIN_N_SAMPLES", "MIN_SEQ_LEN",
    "PROTOCOL_TRAINED_ROWS",
    "T3_ABOVE_FRACTION", "T3_ABOVE_OVER_BELOW",
    "baseline_seed_specs", "classify_quadrant", "classify_tier", "classify_verdict",
    "inert_cls", "prescreen", "probe_scope", "propose_slice",
    "summarize_dataset", "summarize_task",
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


# =====================================================================
# 换源池调研（results/candidate_survey/PROTOCOL.md）：初筛硬条件 + 判据 v2
# =====================================================================

# §2 初筛硬条件（H1/H2/H3）
MIN_SEQ_LEN = 50        # H1 固定 encoder 适用边界（PenDigits L=8 的 CLS spread SNR 0.39）
MIN_N_SAMPLES = 1000    # H2 由 n_eval≥100 反推（R' 噪声 ≈0.5/√n_eval ≤ 典型 δ 0.05）
MIN_N_CLASSES = 5       # H3 类别子集是任务多样性主力（实测类别 ≈0.65 强 / 通道 ≈0.92 弱）

# §4.2 判据 v2 的 T3：把零假设诊断写进判据
T3_ABOVE_FRACTION = 0.5     # 超出 ±2σ 带的条数 ≥ 总条数的一半
T3_ABOVE_OVER_BELOW = 2.0   # 且 超出 ≥ 2× 跌破（挡住"提升与破坏并存"的假阳）
# §4.1 的测量口径：10 shared CLS × 前 2 任务 = 20 条。测不满就不出判定——这是执行
# 协议而非改判据：对 n=2 出判定时 T3 的"超出过半"退化成"超出 ≥1 条"，一条运气好的
# CLS 就能判 usable（PTB-XL 因 L² 代价被停机时正好只剩 2 条，是真实场景）。
PROTOCOL_TRAINED_ROWS = 20


def prescreen(
    *, n_samples: int, seq_len: int, n_classes: int
) -> tuple[bool, str, list[str]]:
    """§2 初筛：→ (是否进基线测量, 建议角色, 未过条件说明)。

    K<MIN_N_CLASSES 不是淘汰而是降 **reserve**：这类候选每条基线只要 0.2 s，测了能
    独立回答"贴顶是不是普遍现象"，但它只能靠通道/样本子集造多样性——UEA 的 2 类
    EEG 源（SCP1 切出 2 任务、SCP2 3 个、MotorImagery 4 个）全部判 inert。
    """
    why: list[str] = []
    if seq_len < MIN_SEQ_LEN:
        why.append(f"L={seq_len} < {MIN_SEQ_LEN}（encoder 适用边界）")
    if n_samples < MIN_N_SAMPLES:
        why.append(f"N={n_samples} < {MIN_N_SAMPLES}（n_eval 不足 100，R' 噪声超 δ）")
    hard_fail = bool(why)
    if n_classes < MIN_N_CLASSES:
        why.append(f"K={n_classes} < {MIN_N_CLASSES}（类别子集撑不起任务多样性）→ reserve")
    if hard_fail:
        return False, "reject", why
    return True, ("source" if n_classes >= MIN_N_CLASSES else "reserve"), why


MIN_N_EVAL = 100          # H2 的精确版：R' 噪声 ≈0.5/√n_eval ≤ 典型 δ 0.05
_EVAL_RATIO = 0.3         # labels.yaml: eval_ratio
_TASK_SAMPLE_CAP = 1024   # source_pool.yaml: slice.n_samples（每任务采样上限）
_MAX_TASKS_PER_DATASET = 16   # 与 source_pool.yaml 现有上限一致（SpokenArabicDigits/CT）


def propose_slice(*, n_samples: int, n_classes: int, n_channels: int) -> dict:
    """给候选提一套切片档（类别子集范围 + 任务数），并估每任务 n_eval。

    这是 PROTOCOL §5 第 2 条"按实际切片档核算 n_eval ≥ 100"的实现，也是步骤 2/3
    成本估算的输入。**只是规划建议**——真正的注册表在步骤 3 判完 usable 之后才写。

    类别档取 [0.4K, 0.8K]（刻意 off-ceiling：类别越多任务越难，避免 R' 贴天花板；
    与 source_pool.yaml 现有档同型，如 K=10→[6,10]、K=26→[8,14] 都落在这个带里）。
    """
    from math import comb

    lo = max(2, round(n_classes * 0.4))
    hi = min(n_classes, max(lo + 1, round(n_classes * 0.8)))
    n_combos = sum(comb(n_classes, k) for k in range(lo, hi + 1))

    # 每任务样本 ≈ N × 该档平均类占比，再受采样上限截断
    frac = (lo + hi) / 2.0 / n_classes
    task_n = min(_TASK_SAMPLE_CAP, n_samples * frac)
    n_eval = int(task_n * _EVAL_RATIO)

    planned = min(_MAX_TASKS_PER_DATASET, n_combos)
    return {
        "n_classes_range": (lo, hi),
        "n_combos": n_combos,
        "planned_tasks": planned,
        "n_channels_range": (max(1, (n_channels + 1) // 2), n_channels),
        "task_n_est": int(task_n),
        "n_eval_est": n_eval,
        "n_eval_ok": n_eval >= MIN_N_EVAL,
        "rprime_noise_est": 0.5 / (n_eval ** 0.5) if n_eval else float("inf"),
    }


def classify_verdict(
    *,
    baseline: float,
    best_gain: Optional[float],
    n_above: int,
    n_below: int,
    n_trained: int,
) -> str:
    """§4.2 判据 v2：ceiling / usable / inert / unknown。

    与 `classify_quadrant`（旧版）的差别是 **T3**——旧版只要 `best_gain ≥ δ` 就判
    usable，而 `best_gain` 是"N 条 CLS 取最大"、天然向上偏。UEA 上因此放行了
    PhonemeSpectra(2↑/7↓) 与 Cricket(1↑/8↓)：它们的增益全由选择偏差撑着。
    新判据要求训练后真的有**过半**条数跑出随机初始化自身的 ±2σ 带，且**超出的
    比跌破的多一倍以上**（挡住 SpokenArabicDigits 那种提升与破坏并存的情形）。

    旧版保留不动（`classify_quadrant`），因为 baseline_probe 的历史 analysis.json
    按它生成——两套判据并存、各自标注出处，比就地改写更可审计。
    """
    if classify_tier(baseline) == "ceiling":
        return "ceiling"          # T1 不过：贴顶直接出局，不看增益
    if best_gain is None or n_trained < PROTOCOL_TRAINED_ROWS:
        return "unknown"          # 没有配对训练数据、或测不满 §4.1 的 20 条，不臆断
    if best_gain < BEST_GAIN_FLOOR:
        return "inert"            # T2 不过：空间在，但没有 CLS 用得上
    if n_above < T3_ABOVE_FRACTION * n_trained:
        return "inert"            # T3 前半：显著超出的条数不过半
    if n_above < T3_ABOVE_OVER_BELOW * n_below:
        return "inert"            # T3 后半：提升没有压过破坏
    return "usable"


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
    # 判据 v2 与旧 quadrant **并列输出**：baseline_probe 的历史 analysis.json 按
    # quadrant 生成，覆盖它会让旧结果不可复算；两套各自标注出处更可审计。
    out["verdict"] = classify_verdict(
        baseline=mean, best_gain=out["best_gain"],
        n_above=out["n_trained_above_null"], n_below=out["n_trained_below_null"],
        n_trained=out["n_trained"],
    ) if vals else "unknown"
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
        # 数据集级 verdict：零假设计数按**跨任务累加**（口径 §4.1 的 20 条本就是
        # "前 2 任务 × 10 CLS"合起来算的），best_gain 仍取跨任务最大（宽松侧）。
        "verdict": classify_verdict(
            baseline=mean, best_gain=best_gain_max,
            n_above=sum(t.get("n_trained_above_null", 0) for t in tasks.values()),
            n_below=sum(t.get("n_trained_below_null", 0) for t in tasks.values()),
            n_trained=sum(t.get("n_trained", 0) for t in tasks.values()),
        ),
    }
