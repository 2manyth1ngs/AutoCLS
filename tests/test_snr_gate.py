"""step0 SNR 预探针的纯分析核心（铁律 5 先测后写；从 probe_route_a.analyze 抽取）。

被测纯函数（无 GPU/训练，用合成 R' 记录）：
- compute_snr_stats：{task: {cls: {seed: R'}}} → 种子噪声 σ、CLS 区分 spread、SNR、
  成对符号稳定率、建议 δ(≈3σ)、（可选）R' 与 CLS 质量单调性。
- decide：SNR≥1 且符号稳定 → 通过；SNR<1 再分天花板/失配——判据不是 R̄ 高低，
  而是"加难重探 SNR 是否恢复"（smoke 实证：NATOPS R̄0.94 贴顶该救，PenDigits
  R̄0.98 也贴顶却是真失配，单看 R̄ 区分不了）。
- escalate_range / ceiling_escalation：天花板加难重探（≤2 轮）纯控制流，
  跑 R' 的 run_round 由调用方注入（step0 接 GPU，测试注入 mock）。

判据与探针同款（results/route_a_probe/REPORT.md §3/§4）。
"""
import numpy as np
import pytest

from autocls.snr_gate import (
    VERDICT_FAIL_MISMATCH,
    VERDICT_RETRY_CEILING,
    ceiling_escalation,
    compute_snr_stats,
    decide,
    escalate_range,
    is_range_maxed,
)


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
# 贴顶无区分：R' 全挤在天花板附近，spread < 噪声 → SNR<1。单看这份数据分不出
# 天花板型（加难可救，NATOPS 式）还是失配型（加难也不恢复，PenDigits 式）——
# decide 只能给 retry_ceiling（未触顶）或 fail_mismatch（已触顶）。
FLAT = {
    "PenDigits_00": {
        "a": {0: 0.982, 1: 0.975, 2: 0.990},   # 抖动 ~噪声，均值几乎相同
        "b": {0: 0.978, 1: 0.986, 2: 0.972},
        "c": {0: 0.985, 1: 0.973, 2: 0.988},
        "d": {0: 0.980, 1: 0.989, 2: 0.974},
    }
}


def _flat_high(seeds=(0, 1, 2)):
    """贴顶且 CLS 无区分：R̄ 高、spread≈0（天花板 or 触顶失配，靠加难重探区分）。"""
    return {"t0": {f"c{i}": {s: 0.97 + 0.001 * ((i + s) % 2) for s in seeds} for i in range(5)}}


def _flat_mid(seeds=(0, 1, 2)):
    """未贴顶且 CLS 无区分：R̄ 中段、spread≈0——非天花板问题，直接失配（踢）。"""
    return {"t0": {f"c{i}": {s: 0.55 + 0.002 * ((i + s) % 2) for s in seeds} for i in range(5)}}


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


def test_decide_flat_high_not_passed():
    # 贴顶 SNR<1：不通过，但默认（未触顶）走 retry_ceiling 而非直接踢
    v = decide(compute_snr_stats(FLAT))
    assert v.passed is False
    assert v.label == VERDICT_RETRY_CEILING
    assert "SNR" in v.reason


def test_decide_respects_snr_floor():
    # 阈值可调：把 floor 抬到很高，好源也判不过（阈值语义正确）
    v = decide(compute_snr_stats(GOOD), snr_floor=99.0)
    assert v.passed is False


# ---- 天花板/失配区分（Part A-1）：判据 = "加难重探 SNR 是否恢复"，非 R̄ 高低 ----


def test_rprime_grand_mean_is_mean_over_all_runs():
    stats = compute_snr_stats(GOOD)
    vals = [v for cm in GOOD.values() for sm in cm.values() for v in sm.values()]
    assert stats.rprime_grand_mean == pytest.approx(np.mean(vals), rel=1e-6)


def test_ceiling_when_high_and_range_not_maxed():
    # NATOPS smoke 实测模式：R̄0.94 贴顶、SNR0.73——该救（上调 range 重探），勿踢
    v = decide(compute_snr_stats(_flat_high()), ceiling_thresh=0.90, range_maxed=False)
    assert v.action == "retry_ceiling" and v.label == VERDICT_RETRY_CEILING
    assert v.passed is False


def test_mismatch_when_high_but_range_maxed():
    # PenDigits 实测模式：R̄0.98 也贴顶，但已在最难可行 range 下仍 SNR<1 → 真失配
    v = decide(compute_snr_stats(_flat_high()), ceiling_thresh=0.90, range_maxed=True)
    assert v.action == "kick" and v.label == VERDICT_FAIL_MISMATCH


def test_mismatch_when_mid_and_flat():
    # 未贴顶的平坦（R̄~0.55）：不是任务过易，是区分被噪声淹没 → 直接踢
    v = decide(compute_snr_stats(_flat_mid()), ceiling_thresh=0.90, range_maxed=False)
    assert v.action == "kick" and v.label == VERDICT_FAIL_MISMATCH


def test_decide_pass_action_is_pass():
    assert decide(compute_snr_stats(GOOD)).action == "pass"


# ---- escalate_range：加难 = lo/hi 各 +2、封顶 n_classes（NATOPS [3,6]/6 类仍可
#      经 [5,6]→[6,6] 两轮加难；2 类源 [2,2] 天然触顶） ----


def test_escalate_range_raises_both_ends():
    assert escalate_range((3, 6), 10) == (5, 8)
    assert escalate_range((5, 8), 10) == (7, 10)


def test_escalate_range_caps_at_n_classes():
    assert escalate_range((3, 6), 6) == (5, 6)
    assert escalate_range((5, 6), 6) == (6, 6)


def test_is_range_maxed():
    assert is_range_maxed((2, 2), 2) is True      # 2 类源：无更难档
    assert is_range_maxed((6, 6), 6) is True
    assert is_range_maxed((3, 6), 6) is False     # NATOPS：还能加难
    assert is_range_maxed((6, 10), 10) is False


# ---- ceiling_escalation：≤2 轮加难重探纯控制流（run_round 注入，无 GPU） ----


def test_ceiling_escalation_recovers():
    # 天花板型 mock：base 贴顶、加难一轮后区分度恢复 → 判可用，记录恢复 range
    calls = []

    def run_round(r, rng):
        calls.append((r, rng))
        return compute_snr_stats(GOOD)   # 加难后 SNR 恢复

    out = ceiling_escalation(compute_snr_stats(_flat_high()), (3, 6), 10, run_round)
    assert out.verdict.passed is True
    assert out.n_classes_range == (5, 8)          # 阶段2 造 label 应沿用的更难档
    assert calls == [(1, (5, 8))]
    assert len(out.rounds) == 1


def test_ceiling_escalation_gives_up_after_max_rounds():
    # 失配型 mock：加难两轮仍贴顶不恢复 → 踢入 reserve 候选
    def run_round(r, rng):
        return compute_snr_stats(_flat_high())    # 加难也不恢复

    out = ceiling_escalation(compute_snr_stats(_flat_high()), (3, 6), 10, run_round)
    assert out.verdict.action == "kick"
    assert out.verdict.label == VERDICT_FAIL_MISMATCH
    assert [r.n_classes_range for r in out.rounds] == [(5, 8), (7, 10)]


def test_ceiling_escalation_skips_when_base_range_maxed():
    # 2 类源贴顶：无更难档可试，base 即触顶 → 直接失配，零重探轮
    def run_round(r, rng):  # pragma: no cover - 不应被调用
        raise AssertionError("range 已触顶不应重探")

    out = ceiling_escalation(compute_snr_stats(_flat_high()), (2, 2), 2, run_round)
    assert out.verdict.label == VERDICT_FAIL_MISMATCH
    assert out.rounds == ()


def test_ceiling_escalation_passthrough_when_base_passes():
    out = ceiling_escalation(compute_snr_stats(GOOD), (3, 6), 10, lambda r, rng: None)
    assert out.verdict.passed is True
    assert out.n_classes_range == (3, 6)
    assert out.rounds == ()
