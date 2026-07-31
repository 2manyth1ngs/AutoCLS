"""SNR 预探针的纯分析核心（step0 与 probe 共用；无 GPU/训练依赖，可完整单测）。

从 scripts/probe_route_a.py::analyze 抽取，判据同款（results/route_a_probe/REPORT.md
§3/§4）。回答两个问题：

1. **CLS 是否可分**（风险2/短序列失配）：CLS 区分 spread（各 CLS 的 seed 均值 R' 的
   std）相对种子噪声 σ 的信噪比 SNR。SNR<1 = CLS 区分被噪声淹没（PenDigits L8 惰性
   模式，R' 与 CLS 质量零相关），只产噪声 label，剔除或降权。
2. **δ 该多大**（风险1/R' 可靠性）：种子噪声 σ 决定 δ（探针 §2.2：δ 取噪声上界略高
   处 ≈3σ）。

输入统一结构 `rprime[task_id][cls_key][seed] = R'`（R' 可为 NaN 表失败）。方向无关
（只测数值离散度，不造 y）——铁律 8 的 is_better 只在真正造 label 时用。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Optional

import numpy as np

RPrimeByTask = dict[str, dict[str, dict[int, float]]]

# decide 判据标签。SNR<1 的旧一刀切 fail_snr 已拆成天花板/失配两型（Part A-1）：
# smoke 实证 NATOPS R̄0.94 贴顶（任务过易该救）与 PenDigits R̄0.98 贴顶（架构失配
# 该踢）单看 R̄ 分不出——唯一判据是"加难重探后 SNR 是否恢复"。
VERDICT_PASS = "pass"
VERDICT_RETRY_CEILING = "retry_ceiling"    # 贴顶且 range 未触顶：加难重探（勿踢）
VERDICT_FAIL_MISMATCH = "fail_mismatch"    # 最难可行 range 下仍 SNR<1：真失配，踢
VERDICT_FAIL_UNSTABLE = "fail_unstable"
VERDICT_INSUFFICIENT = "insufficient"

# 天花板加难重探最多轮数（用户拍板：step0 内自动加难 ≤2 轮，不踢出去手动重跑）
MAX_CEILING_ROUNDS = 2


@dataclass(frozen=True)
class SnrStats:
    n_tasks: int
    n_fail: int
    rprime_std_mean: float        # 种子噪声 σ（mean over (task,cls) 跨 seed std）
    rprime_std_p90: float
    cls_spread_std_mean: float    # CLS 区分信号（mean over task 的 per-cls seed 均值 std）
    snr: float                    # spread / noise
    pair_sign_stability: float    # 成对 (R'_a-R'_b) 跨 seed 符号稳定率
    suggested_delta: float        # ≈3σ（探针 §2.2）
    monotonicity: float | None    # Spearman(R̄, -har_rank)，无 har_rank 时 None
    rprime_grand_mean: float = float("nan")  # 所有有限 (task,cls,seed) R' 总均值（贴顶判据）


@dataclass(frozen=True)
class Verdict:
    passed: bool
    label: str
    reason: str
    action: str = "pass"          # pass | retry_ceiling | kick | kfold


def _std(vals: np.ndarray) -> float:
    return float(vals.std(ddof=1)) if len(vals) >= 2 else float("nan")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 秩相关（纯 numpy：秩化后 Pearson）——避免引 scipy 依赖。"""
    if len(a) < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def compute_snr_stats(
    rprime: RPrimeByTask,
    *,
    har_rank: dict[str, int] | None = None,
    delta_ref: float = 0.02,
) -> SnrStats:
    """聚合合成/实测 R' → 数据集级 SNR 判据（语义见模块 docstring）。"""
    rprime_stds: list[float] = []           # ① 每 (task,cls) 跨 seed std
    spread_stds: list[float] = []           # ② 每 task 的 per-cls seed 均值 spread
    sign_stable: list[float] = []           # ③ 成对符号稳定
    cls_mean_accum: dict[str, list[float]] = {}   # 跨 task 汇总各 cls 的 seed 均值（单调性）
    n_fail = 0

    for clsmap in rprime.values():
        cls_seed_mean: dict[str, float] = {}
        seeds_all = sorted({s for sm in clsmap.values() for s in sm})
        for key, sm in clsmap.items():
            vals = np.array([sm[s] for s in sorted(sm)], dtype=float)
            n_fail += int(np.isnan(vals).sum())
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 2:
                rprime_stds.append(_std(vals))
            if len(vals) >= 1:
                cls_seed_mean[key] = float(vals.mean())
                cls_mean_accum.setdefault(key, []).append(float(vals.mean()))
        means = np.array([m for m in cls_seed_mean.values() if np.isfinite(m)])
        if len(means) >= 2:
            spread_stds.append(_std(means))
        # 成对符号稳定：同一 (task) 下每对 CLS 跨共享 seed 的差值符号是否恒定
        keys = list(clsmap.keys())
        for a, b in combinations(keys, 2):
            diffs = [clsmap[a][s] - clsmap[b][s] for s in seeds_all
                     if s in clsmap[a] and s in clsmap[b]
                     and np.isfinite(clsmap[a][s]) and np.isfinite(clsmap[b][s])]
            signs = np.sign([d for d in diffs if abs(d) > 1e-12])
            if len(signs) >= 2:
                sign_stable.append(float(np.all(signs == signs[0])))

    grand = [v for cm in rprime.values() for sm in cm.values()
             for v in sm.values() if np.isfinite(v)]
    rprime_grand_mean = float(np.mean(grand)) if grand else float("nan")
    rprime_std_mean = float(np.mean(rprime_stds)) if rprime_stds else float("nan")
    rprime_std_p90 = float(np.percentile(rprime_stds, 90)) if rprime_stds else float("nan")
    cls_spread_std_mean = float(np.mean(spread_stds)) if spread_stds else float("nan")
    snr = (cls_spread_std_mean / rprime_std_mean
           if rprime_std_mean and rprime_std_mean == rprime_std_mean else float("nan"))
    pair_sign = float(np.mean(sign_stable)) if sign_stable else float("nan")
    suggested_delta = round(3 * rprime_std_mean, 3) if rprime_std_mean == rprime_std_mean else float("nan")

    monotonicity: float | None = None
    if har_rank is not None:
        keys = [k for k in cls_mean_accum if k in har_rank]
        if len(keys) >= 2:
            rbar = np.array([np.mean(cls_mean_accum[k]) for k in keys])
            neg_rank = np.array([-har_rank[k] for k in keys], dtype=float)
            monotonicity = _spearman(rbar, neg_rank)

    return SnrStats(
        n_tasks=len(rprime), n_fail=n_fail,
        rprime_std_mean=rprime_std_mean, rprime_std_p90=rprime_std_p90,
        cls_spread_std_mean=cls_spread_std_mean, snr=snr,
        pair_sign_stability=pair_sign, suggested_delta=suggested_delta,
        monotonicity=monotonicity, rprime_grand_mean=rprime_grand_mean,
    )


def decide(
    stats: SnrStats,
    *,
    snr_floor: float = 1.0,
    sign_floor: float = 0.7,
    ceiling_thresh: float = 0.90,
    range_maxed: bool = False,
) -> Verdict:
    """SNR 闸判定：SNR≥floor 且符号稳定 → 通过；SNR<floor 再分天花板/失配。

    天花板/失配区分（Part A-1）：R̄ 高低本身区分不了两者（NATOPS R̄0.94 该救、
    PenDigits R̄0.98 该踢），唯一判据是"加难重探 SNR 是否恢复"——本函数只做单轮
    静态判定，重探循环见 ceiling_escalation：
    - SNR<floor 且 R̄≥ceiling_thresh 且 range 未触顶 → retry_ceiling（上调
      n_classes_range 切更难任务重探，勿踢）；
    - range 已触顶（最难可行档仍贴顶）或 R̄ 未贴顶的平坦 → fail_mismatch（踢）；
    - SNR 够但符号不稳：R' 噪声大到成对方向翻转 → fail_unstable（须 k 折/大 δ）。

    range_maxed 由调用方给出："无更难档可试"（含重探轮数用尽），见 is_range_maxed。
    """
    if stats.snr != stats.snr:  # NaN
        return Verdict(False, VERDICT_INSUFFICIENT,
                       "数据不足：无法计算 SNR（种子/CLS 太少）", action="kick")
    if stats.snr < snr_floor:
        near_ceiling = (stats.rprime_grand_mean == stats.rprime_grand_mean
                        and stats.rprime_grand_mean >= ceiling_thresh)
        if near_ceiling and not range_maxed:
            return Verdict(
                False, VERDICT_RETRY_CEILING,
                f"SNR={stats.snr:.2f}<{snr_floor} 但 R̄={stats.rprime_grand_mean:.3f}"
                f"≥{ceiling_thresh} 贴顶：任务过易、CLS 无从区分；"
                "上调 n_classes_range 切更难任务重探（勿踢）",
                action="retry_ceiling",
            )
        return Verdict(
            False, VERDICT_FAIL_MISMATCH,
            f"SNR={stats.snr:.2f}<{snr_floor}"
            + (f"，且已达最难可行 range 仍贴顶（R̄={stats.rprime_grand_mean:.3f}）："
               "架构失配（PenDigits 式）" if range_maxed and near_ceiling
               else f"，R̄={stats.rprime_grand_mean:.3f} 未贴顶：CLS 区分被噪声淹没")
            + "；踢入 reserve 候选",
            action="kick",
        )
    if stats.pair_sign_stability == stats.pair_sign_stability and stats.pair_sign_stability < sign_floor:
        return Verdict(
            False, VERDICT_FAIL_UNSTABLE,
            f"符号稳定率={stats.pair_sign_stability:.2f}<{sign_floor}："
            f"成对方向被噪声翻转（须 k 折降噪或大 δ）",
            action="kfold",
        )
    return Verdict(
        True, VERDICT_PASS,
        f"SNR={stats.snr:.2f}≥{snr_floor} 且符号稳定（可用；建议 δ≈{stats.suggested_delta}）",
        action="pass",
    )


# ---------------------------------------------------------------------------
# 天花板加难重探（Part A-1）：≤MAX_CEILING_ROUNDS 轮纯控制流；跑 R' 由调用方注入
# ---------------------------------------------------------------------------


def escalate_range(n_classes_range: tuple[int, int], n_classes: int) -> tuple[int, int]:
    """加难一档：lo/hi 各 +2、封顶 n_classes（hi 按方案 min(hi+2, n_classes)；lo 同步
    上调是补全——低类数源如 NATOPS [3,6]/6 类的 hi 已在顶，仅提 hi 无法加难，提 lo
    才能逼出 [5,6]→[6,6] 的更难任务，方案 B-7"救回 NATOPS/ERing"以此为前提）。"""
    lo, hi = n_classes_range
    new_hi = min(hi + 2, n_classes)
    new_lo = min(lo + 2, new_hi)
    return (new_lo, new_hi)


def is_range_maxed(n_classes_range: tuple[int, int], n_classes: int) -> bool:
    """无更难档可试（加难产出同一 range）：2 类源 [2,2] 天然触顶；[6,6]/6 类同。"""
    return escalate_range(tuple(n_classes_range), n_classes) == tuple(n_classes_range)


@dataclass(frozen=True)
class RetryRound:
    """一轮加难重探的记录（round 从 1 起；base 轮不在此列）。"""

    round: int
    n_classes_range: tuple[int, int]
    stats: SnrStats
    verdict: Verdict


@dataclass(frozen=True)
class RetryOutcome:
    """天花板重探终局：verdict 为最终判定；n_classes_range 为最终采用档——
    passed 时即阶段2 造 label 应沿用的 recommended range。"""

    verdict: Verdict
    n_classes_range: tuple[int, int]
    rounds: tuple[RetryRound, ...]


def ceiling_escalation(
    base_stats: SnrStats,
    base_range: tuple[int, int],
    n_classes: int,
    run_round: Callable[[int, tuple[int, int]], Optional[SnrStats]],
    *,
    max_rounds: int = MAX_CEILING_ROUNDS,
    snr_floor: float = 1.0,
    sign_floor: float = 0.7,
    ceiling_thresh: float = 0.90,
) -> RetryOutcome:
    """天花板加难重探循环（纯控制流，无 GPU 依赖，可注入 mock 完整单测）。

    判 retry_ceiling 时逐轮 escalate_range 并调 run_round(round_idx, new_range)
    重跑 R' 得新 stats，≤max_rounds 轮：
    - 某轮 SNR 恢复 → 天花板型，判可用，n_classes_range 记录恢复所需档
      （阶段2 造 label 沿用）；
    - 轮数用尽或 range 触顶仍 SNR<1 → 失配型（PenDigits 模式），判踢。
    run_round 由调用方注入（step0 接真实切片+训练；测试注入合成 stats）。
    """
    cur_range = tuple(base_range)
    verdict = decide(base_stats, snr_floor=snr_floor, sign_floor=sign_floor,
                     ceiling_thresh=ceiling_thresh,
                     range_maxed=is_range_maxed(cur_range, n_classes))
    rounds: list[RetryRound] = []
    while verdict.action == "retry_ceiling" and len(rounds) < max_rounds:
        r = len(rounds) + 1
        cur_range = escalate_range(cur_range, n_classes)
        stats = run_round(r, cur_range)
        # 触顶或轮数用尽后仍 SNR<1 即失配——最后一轮判定必须以 range_maxed=True 收口
        maxed = (r >= max_rounds) or is_range_maxed(cur_range, n_classes)
        verdict = decide(stats, snr_floor=snr_floor, sign_floor=sign_floor,
                         ceiling_thresh=ceiling_thresh, range_maxed=maxed)
        rounds.append(RetryRound(round=r, n_classes_range=cur_range,
                                 stats=stats, verdict=verdict))
    return RetryOutcome(verdict=verdict, n_classes_range=cur_range, rounds=tuple(rounds))
