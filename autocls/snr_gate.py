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

import numpy as np

RPrimeByTask = dict[str, dict[str, dict[int, float]]]

# decide 判据标签
VERDICT_PASS = "pass"
VERDICT_FAIL_SNR = "fail_snr"
VERDICT_FAIL_UNSTABLE = "fail_unstable"
VERDICT_INSUFFICIENT = "insufficient"


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


@dataclass(frozen=True)
class Verdict:
    passed: bool
    label: str
    reason: str


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
        monotonicity=monotonicity,
    )


def decide(stats: SnrStats, *, snr_floor: float = 1.0, sign_floor: float = 0.7) -> Verdict:
    """SNR 闸判定（探针同款）：SNR≥floor 且符号稳定 → 通过；否则剔除/降权。

    - SNR<floor：CLS 区分被噪声淹没（短序列/小样本失效，PenDigits 模式）→ fail_snr。
    - SNR 够但符号不稳：R' 噪声大到成对方向都翻转 → fail_unstable（须 k 折或大 δ）。
    """
    if stats.snr != stats.snr:  # NaN
        return Verdict(False, VERDICT_INSUFFICIENT, "数据不足：无法计算 SNR（种子/CLS 太少）")
    if stats.snr < snr_floor:
        return Verdict(
            False, VERDICT_FAIL_SNR,
            f"SNR={stats.snr:.2f}<{snr_floor}：CLS 区分被噪声淹没（剔除或降权）",
        )
    if stats.pair_sign_stability == stats.pair_sign_stability and stats.pair_sign_stability < sign_floor:
        return Verdict(
            False, VERDICT_FAIL_UNSTABLE,
            f"符号稳定率={stats.pair_sign_stability:.2f}<{sign_floor}："
            f"成对方向被噪声翻转（须 k 折降噪或大 δ）",
        )
    return Verdict(
        True, VERDICT_PASS,
        f"SNR={stats.snr:.2f}≥{snr_floor} 且符号稳定（可用；建议 δ≈{stats.suggested_delta}）",
    )
