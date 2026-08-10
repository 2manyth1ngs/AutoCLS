"""iters 工作点扫描的判定逻辑单测（results/iters_workpoint/PROTOCOL.md §4，先于实现）。

判定全部是纯函数（吃 arm 摘要，不碰缓存/数据集），阈值逐条对协议：
- 峰 = best_gain 的 argmax（判据 2），网格端点截尾；
- 峰值一致性 max/min ≤2 一致 / ≥4 不一致 / 之间或截尾 → 存疑（判据 3）；
- 翻盘 稳（↑≥11 或相邻两点过）/ 弱（单点恰 10/20）（判据 4）；
- 无工作点 = 净收益维 (a) 与 label 产量维 (b) 都空（判据 5，用户拍板补 b 维）。
"""
import math

import pytest

from autocls.workpoint import (
    arm_epoch,
    classify_flip,
    confirm_verdict,
    destruction_participation,
    find_peak,
    no_workpoint_verdict,
    peak_consistency,
    steps_per_epoch,
    wilson_interval,
)


# ---------------------------------------------------------------- epoch 换算
class TestEpochConversion:
    @pytest.mark.parametrize("n_train,expected", [
        (717, 6), (716, 6), (718, 6),   # 候选池切片任务实测值
        (360, 3), (289, 3),             # Handwriting 前 2 任务实测值
        (128, 1), (129, 2), (1, 1),     # 边界：恰好整除 / 多 1 / 最小
        (7352, 58),                     # M1 HAR 基准（290 iters = 5.0 ep 的出处）
    ])
    def test_steps_per_epoch(self, n_train, expected):
        assert steps_per_epoch(n_train) == expected

    def test_arm_epoch(self):
        assert arm_epoch(290, 58) == pytest.approx(5.0)
        assert arm_epoch(290, 6) == pytest.approx(48.333, abs=1e-3)
        assert arm_epoch(30, 3) == pytest.approx(10.0)

    def test_steps_per_epoch_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            steps_per_epoch(0)


# ---------------------------------------------------------------- 峰定位（判据 2）
class TestFindPeak:
    def test_interior_peak(self):
        # Handwriting 既有三点：30→+0.249, 290→+0.397, 2320→+0.372 → 峰 290，非截尾
        p = find_peak({30: 0.249, 290: 0.397, 2320: 0.372})
        assert p["iters"] == 290 and p["value"] == 0.397 and p["censored"] is None

    def test_right_edge_censored(self):
        # WISDM 既有三点单调涨：峰在网格右端 → 截尾，真峰可能在网格外
        p = find_peak({30: 0.038, 290: 0.074, 2320: 0.100})
        assert p["iters"] == 2320 and p["censored"] == "right"

    def test_left_edge_censored(self):
        p = find_peak({30: 0.30, 290: 0.20, 2320: 0.10})
        assert p["iters"] == 30 and p["censored"] == "left"

    def test_tie_prefers_smaller_iters(self):
        # 平局取更小 iters（更便宜的工作点；确定性）
        p = find_peak({30: 0.10, 290: 0.10, 2320: 0.05})
        assert p["iters"] == 30

    def test_none_values_skipped(self):
        # 未测/测不满的臂（best_gain=None）不参与 argmax；且它不算"测过的网格点"
        # ——峰落在有效点的边缘仍是截尾（290 以下的曲线未知）
        p = find_peak({30: None, 290: 0.07, 2320: 0.05})
        assert p["iters"] == 290 and p["censored"] == "left"
        assert find_peak({30: None}) is None

    def test_single_point_is_both_censored(self):
        # 只有一个点谈不上峰：两侧都截尾，报告层必须按截尾处理
        p = find_peak({290: 0.07})
        assert p["censored"] == "both"


# ---------------------------------------------------------------- 峰一致性（判据 3）
class TestPeakConsistency:
    def test_consistent_ratio_le_2(self):
        # 精确值：290 iters 在 6 与 3 steps/epoch 下 = 48.33 与 96.67 ep，比恰 2.0
        # （四舍五入的 96.7/48.3=2.002 会误落"存疑"——真实比较必须用精确 epoch）
        v = peak_consistency({"A": (290 / 6, None), "B": (290 / 3, None)})
        assert v["verdict"] == "consistent"
        assert v["ratio"] == pytest.approx(2.0)

    def test_inconsistent_ratio_ge_4(self):
        v = peak_consistency({"A": (5.0, None), "B": (96.7, None)})
        assert v["verdict"] == "inconsistent"

    def test_between_is_uncertain(self):
        v = peak_consistency({"A": (20.0, None), "B": (48.3, None)})  # 比 2.4
        assert v["verdict"] == "uncertain"

    def test_any_censored_is_uncertain(self):
        # 协议原文：截尾峰参与 → 存疑（即便比值本身 ≤2 或 ≥4）
        v = peak_consistency({"A": (48.3, "right"), "B": (96.7, None)})
        assert v["verdict"] == "uncertain"
        v = peak_consistency({"A": (5.0, "right"), "B": (96.7, None)})
        assert v["verdict"] == "uncertain"

    def test_single_dataset_no_verdict(self):
        assert peak_consistency({"A": (48.3, None)})["verdict"] == "insufficient"


# ---------------------------------------------------------------- 翻盘（判据 4）
def _arm(usable=False, n_above=0):
    return {"usable": usable, "n_above": n_above}


class TestClassifyFlip:
    def test_no_flip(self):
        v = classify_flip({30: _arm(), 290: _arm(n_above=9)})
        assert v["flip"] == "none" and v["at"] == []

    def test_weak_single_point_exactly_10(self):
        v = classify_flip({30: _arm(), 60: _arm(usable=True, n_above=10), 290: _arm()})
        assert v["flip"] == "weak" and v["at"] == [60]

    def test_solid_by_margin_11(self):
        v = classify_flip({60: _arm(usable=True, n_above=11), 290: _arm()})
        assert v["flip"] == "solid"

    def test_solid_by_adjacent_pair(self):
        # 相邻网格点都过 T3（各自只有 10 也算稳：两点独立撞线概率平方级）
        v = classify_flip({
            30: _arm(usable=True, n_above=10), 60: _arm(usable=True, n_above=10),
            290: _arm(),
        })
        assert v["flip"] == "solid" and v["at"] == [30, 60]

    def test_two_nonadjacent_weak_points_stay_weak(self):
        # 非相邻的两个 10/20 点不算稳（协议只认"相邻两点"或"单点 ≥11"）
        v = classify_flip({
            30: _arm(usable=True, n_above=10), 60: _arm(),
            290: _arm(usable=True, n_above=10),
        })
        assert v["flip"] == "weak" and v["at"] == [30, 290]


# ---------------------------------------------------------------- 无工作点（判据 5）
def _narm(realized, t2=False, t3=False):
    return {"realized_gain": realized, "T2": t2, "T3": t3}


class TestNoWorkpoint:
    QUANTUM = 1 / 307  # WISDM n_eval=307 → 一个 ACC 量子

    def test_confirmed_both_dims_empty(self):
        arms = {30: _narm(-0.006), 290: _narm(-0.021, t2=True), 2320: _narm(-0.032, t2=True)}
        v = no_workpoint_verdict(arms, quantum=self.QUANTUM)
        assert v["verdict"] == "no_workpoint"
        assert v["net_gain_dim_empty"] and v["label_dim_empty"]

    def test_label_workpoint_only(self):
        # 用户拍板的 b 维：realized 全负但某点 T2+T3 都过 → 对造 label 是可用的
        arms = {30: _narm(-0.006), 290: _narm(-0.021, t2=True, t3=True)}
        v = no_workpoint_verdict(arms, quantum=self.QUANTUM)
        assert v["verdict"] == "label_workpoint_only"

    def test_has_net_gain(self):
        arms = {30: _narm(0.140, t2=True, t3=True), 290: _narm(0.161, t2=True, t3=True)}
        v = no_workpoint_verdict(arms, quantum=self.QUANTUM)
        assert v["verdict"] == "has_net_gain"

    def test_quantum_tolerance(self):
        # realized = +0.0030 < 一个量子(0.00326) → 仍算净收益维空（数不出一个样本的差）
        arms = {30: _narm(0.0030)}
        v = no_workpoint_verdict(arms, quantum=self.QUANTUM)
        assert v["net_gain_dim_empty"]

    def test_nan_realized_not_counted_as_gain(self):
        arms = {30: _narm(float("nan"))}
        v = no_workpoint_verdict(arms, quantum=self.QUANTUM)
        assert v["net_gain_dim_empty"]


# ------------------------------------------------- Wilson 区间（followup 协议 §1）
class TestWilsonInterval:
    def test_report_values_12_of_20(self):
        # iters_workpoint REPORT §2.1 引用值：12/20 → [0.42, 0.76]（90%，z=1.645）
        lo, hi = wilson_interval(12, 20)
        assert lo == pytest.approx(0.4185, abs=5e-4)
        assert hi == pytest.approx(0.7576, abs=5e-4)

    def test_handwriting_15_of_20_escapes_half(self):
        # REPORT_STEP3 §2：15/20 → [0.57, 0.87]，下界已高于 0.5
        lo, hi = wilson_interval(15, 20)
        assert lo == pytest.approx(0.5678, abs=5e-4)
        assert lo > 0.5

    def test_confirmation_threshold_37_of_60(self):
        # followup 协议预注册门槛：n=60 时脱离 0.5 需 ↑≥37（37→0.511 / 36→0.494）
        assert wilson_interval(37, 60)[0] > 0.5
        assert wilson_interval(36, 60)[0] < 0.5

    def test_degenerate_endpoints(self):
        assert wilson_interval(0, 20)[0] == pytest.approx(0.0)
        assert wilson_interval(20, 20)[1] == pytest.approx(1.0)

    def test_rejects_invalid_counts(self):
        with pytest.raises(ValueError):
            wilson_interval(21, 20)
        with pytest.raises(ValueError):
            wilson_interval(0, 0)


# ------------------------------------- 加测确认层级（followup 协议 §1，用户拍板）
def _confirm(n_above, n_below, n=60, t1=True, t2=True):
    return confirm_verdict(n_above=n_above, n_below=n_below, n_trained=n,
                           t1=t1, t2=t2)


class TestConfirmVerdict:
    def test_tier_a_confirmed_is_pool_eligible(self):
        v = _confirm(37, 3)
        assert v["tier"] == "confirmed" and v["pool_eligible"]
        assert v["t3"] and v["wilson_low"] > 0.5

    def test_tier_b_marginal_not_pool_eligible(self):
        # 用户裁决：B 层（T3 过但 Wilson 含 0.5）不算入池依据
        v = _confirm(36, 0)
        assert v["tier"] == "marginal" and not v["pool_eligible"]

    def test_tier_b_boundary_exactly_half_and_2x(self):
        # T3 边界：30 = 恰 0.5×60 且 恰 2×15 —— 判据原文是 ≥，两处都过
        v = _confirm(30, 15)
        assert v["t3"] and v["tier"] == "marginal"

    def test_tier_c_reversed_below_half(self):
        v = _confirm(29, 0)
        assert v["tier"] == "reversed" and not v["pool_eligible"]

    def test_tier_c_reversed_when_destruction_catches_up(self):
        # ↑=40 但 ↓=21：40 < 2×21 → T3 后半不过 → 反转（60 条优先于 20 条）
        v = _confirm(40, 21)
        assert not v["t3"] and v["tier"] == "reversed"

    def test_t1_t2_failure_is_reversal_even_with_high_above(self):
        assert _confirm(50, 0, t2=False)["tier"] == "reversed"
        assert _confirm(50, 0, t1=False)["tier"] == "reversed"

    def test_empty_arm_is_reversed(self):
        assert _confirm(0, 0, n=0)["tier"] == "reversed"

    def test_uses_prereg_wilson_threshold(self):
        # confirm 的 Wilson 必须与 wilson_interval 同一实现（阈值不许两处漂移）
        v = _confirm(37, 0)
        assert v["wilson_low"] == pytest.approx(wilson_interval(37, 60)[0])


# --------------------------- 破坏参与率（followup 协议 §1；如实报告，不设硬门槛）
class TestDestructionParticipation:
    # 单任务 4 条 CLS：基线 mean=0.75、带半宽 0.04（与 summarize_task 的
    # null_band 同义：strict 破坏 = R' < 0.71，loose = R' < 0.75）
    TASK = {"t0": {"a": 0.80, "b": 0.74, "c": 0.72, "d": 0.60}}
    BASE = {"t0": (0.75, 0.04)}

    def test_counts_on_hand_built_example(self):
        # δ=0.05 存活 5/6 对（bc 差 0.02 不存活）；strict 只有 d（0.60<0.71）
        # 参与 ad/bd/cd = 3/5；loose b/c/d 参与全部 5 对
        v = destruction_participation(self.TASK, self.BASE, delta=0.05)
        assert v["n_pairs"] == 6 and v["n_survive"] == 5
        assert v["n_strict"] == 3 and v["strict_share"] == pytest.approx(0.6)
        assert v["n_loose"] == 5 and v["loose_share"] == pytest.approx(1.0)
        assert v["n_cls"] == 4 and v["n_below_band"] == 1 and v["n_below_mean"] == 3

    def test_pairs_do_not_cross_tasks(self):
        # 配对只在任务内（label 造对的口径）：2 任务 × 2 CLS = 各 1 对，非跨任务 6 对
        per_task = {"t0": {"a": 0.9, "b": 0.7}, "t1": {"a": 0.9, "b": 0.7}}
        base = {"t0": (0.8, 0.02), "t1": (0.8, 0.02)}
        v = destruction_participation(per_task, base, delta=0.05)
        assert v["n_pairs"] == 2 and v["n_survive"] == 2

    def test_exact_delta_survives(self):
        # |ΔR'| 恰等于 δ 算存活（造 label 的 ≥δ 口径）
        v = destruction_participation({"t0": {"a": 0.80, "b": 0.75}},
                                      {"t0": (0.9, 0.01)}, delta=0.05)
        assert v["n_survive"] == 1

    def test_no_survivors_yields_none_shares(self):
        v = destruction_participation({"t0": {"a": 0.75, "b": 0.74}},
                                      {"t0": (0.7, 0.01)}, delta=0.05)
        assert v["n_survive"] == 0
        assert v["strict_share"] is None and v["loose_share"] is None

    def test_failed_rows_excluded(self):
        # r_prime=None/NaN 是"没跑出来"，不参与配对也不计入 CLS 数
        per_task = {"t0": {"a": 0.80, "b": None, "c": float("nan"), "d": 0.60}}
        v = destruction_participation(per_task, {"t0": (0.75, 0.04)}, delta=0.05)
        assert v["n_cls"] == 2 and v["n_pairs"] == 1 and v["n_survive"] == 1
