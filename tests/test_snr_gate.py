"""step0 SNR 预探针的纯分析核心（铁律 5 先测后写；从 probe_route_a.analyze 抽取）。

被测纯函数（无 GPU/训练，用合成 R' 记录）：
- compute_snr_stats：{task: {cls: {seed: R'}}} → 种子噪声 σ、CLS 区分 spread、SNR、
  成对符号稳定率、建议 δ(≈3σ)、（可选）R' 与 CLS 质量单调性。
- decide：SNR≥1 且符号稳定 → 通过；SNR<1（PenDigits 惰性模式）→ 剔除。

判据与探针同款（results/route_a_probe/REPORT.md §3/§4）。
"""
import numpy as np
import pytest

from autocls.snr_gate import compute_snr_stats, decide


def _std(vals):
    return float(np.std(np.array(list(vals), dtype=float), ddof=1))


# 好谱源：4 个 CLS 的 R' 明显分层（spread 大），种子噪声小 → SNR>>1（可用）
GOOD = {
    "PhonemeSpectra_00": {
        "a": {0: 0.70, 1: 0.71, 2: 0.69},   # 好 CLS
        "b": {0: 0.62, 1: 0.63, 2: 0.61},
        "c": {0: 0.55, 1: 0.56, 2: 0.54},
        "d": {0: 0.49, 1: 0.50, 2: 0.48},   # 差 CLS
    }
}
# PenDigits 惰性模式：R' 全挤在天花板附近，spread < 噪声 → SNR<1（剔除）
FLAT = {
    "PenDigits_00": {
        "a": {0: 0.982, 1: 0.975, 2: 0.990},   # 抖动 ~噪声，均值几乎相同
        "b": {0: 0.978, 1: 0.986, 2: 0.972},
        "c": {0: 0.985, 1: 0.973, 2: 0.988},
        "d": {0: 0.980, 1: 0.989, 2: 0.974},
    }
}


# ---- compute_snr_stats：聚合语义 ----

def test_seed_noise_is_mean_of_per_cls_std():
    stats = compute_snr_stats(GOOD)
    expected = np.mean([_std(sm.values()) for sm in GOOD["PhonemeSpectra_00"].values()])
    assert stats.rprime_std_mean == pytest.approx(expected, rel=1e-6)


def test_cls_spread_uses_seed_means():
    stats = compute_snr_stats(GOOD)
    cls_means = [np.mean(list(sm.values())) for sm in GOOD["PhonemeSpectra_00"].values()]
    assert stats.cls_spread_std_mean == pytest.approx(_std(cls_means), rel=1e-6)


def test_snr_is_spread_over_noise():
    stats = compute_snr_stats(GOOD)
    assert stats.snr == pytest.approx(stats.cls_spread_std_mean / stats.rprime_std_mean, rel=1e-6)


def test_good_dataset_snr_above_one():
    assert compute_snr_stats(GOOD).snr > 1.0


def test_flat_dataset_snr_below_one():
    # 贴天花板：spread（≈0）远小于种子噪声 → SNR<1（探针 PenDigits 0.39 模式）
    assert compute_snr_stats(FLAT).snr < 1.0


def test_suggested_delta_is_three_sigma():
    stats = compute_snr_stats(GOOD)
    assert stats.suggested_delta == pytest.approx(round(3 * stats.rprime_std_mean, 3))


def test_sign_stability_high_when_ranking_consistent():
    # 好源里 a>b>c>d 每个种子都成立 → 成对符号完全稳定
    assert compute_snr_stats(GOOD).pair_sign_stability == pytest.approx(1.0)


def test_monotonicity_with_cls_quality_rank():
    # har_rank 越小越好；R̄ 应随质量单调 → Spearman(R̄, -rank) 接近 +1
    har_rank = {"a": 0, "b": 4, "c": 9, "d": 15}
    stats = compute_snr_stats(GOOD, har_rank=har_rank)
    assert stats.monotonicity is not None
    assert stats.monotonicity > 0.9


def test_monotonicity_none_without_ranks():
    assert compute_snr_stats(GOOD).monotonicity is None


def test_counts_failures():
    data = {"T_00": {"a": {0: 0.7, 1: float("nan")}, "b": {0: 0.6, 1: 0.61}}}
    stats = compute_snr_stats(data)
    assert stats.n_fail == 1
    assert stats.n_tasks == 1


# ---- decide：通过/剔除 ----

def test_decide_pass_for_good():
    v = decide(compute_snr_stats(GOOD))
    assert v.passed is True
    assert v.label == "pass"


def test_decide_fail_for_flat_snr():
    v = decide(compute_snr_stats(FLAT))
    assert v.passed is False
    assert v.label == "fail_snr"       # SNR<1：CLS 区分被噪声淹没（剔除或降权）
    assert "SNR" in v.reason


def test_decide_respects_snr_floor():
    # 阈值可调：把 floor 抬到很高，好源也判不过（阈值语义正确）
    v = decide(compute_snr_stats(GOOD), snr_floor=99.0)
    assert v.passed is False
