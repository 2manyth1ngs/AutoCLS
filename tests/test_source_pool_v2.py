"""源池 v2 定稿（scripts/27_source_pool_v2.py）纯逻辑测试。

协议 results/source_pool_v2/PROTOCOL.md §4：δ 存活率（严格 >δ、池化）、
晋级判（锁位/降序/上限/贴顶/死平）、EEG 低权判、靶池三档判、GGS 任务数。
期望值手工推导。
"""
import importlib

import pytest

sp = importlib.import_module("scripts.27_source_pool_v2")


# ---------------------------------------------------------------- δ 存活率
class TestDatasetSurvival:
    def test_single_task_hand_case(self):
        tasks = {"D_00": {"a": 0.5, "b": 0.6, "c": 0.61}}
        s = sp.dataset_survival(tasks, delta=0.05)
        # 配对 |Δ|: ab 0.10✓ ac 0.11✓ bc 0.01✗ → 2/3
        assert s["n_pairs"] == 3
        assert s["survival"] == pytest.approx(2 / 3)
        assert s["max_r"] == pytest.approx(0.61)
        assert s["range"] == pytest.approx(0.11)

    def test_pooled_across_tasks(self):
        tasks = {"D_00": {"a": 0.5, "b": 0.6, "c": 0.61},
                 "D_01": {"a": 0.5, "b": 0.5, "c": 0.5}}   # 全平 0/3
        s = sp.dataset_survival(tasks, delta=0.05)
        assert s["n_pairs"] == 6
        assert s["survival"] == pytest.approx(2 / 6)
        assert s["range"] == pytest.approx(0.11)           # 跨任务全格全距

    def test_strict_inequality_at_delta(self):
        # 二进制精确值（0.5625−0.5=0.0625 逐位等于 δ），钉死严格 > 语义
        tasks = {"D_00": {"a": 0.5, "b": 0.5625}}
        s = sp.dataset_survival(tasks, delta=0.0625)
        assert s["survival"] == 0.0                        # |Δ|==δ 不算存活


# ---------------------------------------------------------------- 晋级判
def _st(surv, max_r=0.9, rng=0.3):
    return {"survival": surv, "max_r": max_r, "range": rng}


class TestPromotionVerdict:
    def test_locks_first_then_backups_desc_cap4(self):
        stats = {"SpokenArabicDigits": _st(0.50),
                 "EthanolConcentration": _st(0.45),
                 "NATOPS": _st(0.55), "HandMovementDirection": _st(0.50),
                 "Cricket": _st(0.48)}
        got = sp.promotion_verdict(stats)
        assert got == ["SpokenArabicDigits", "EthanolConcentration",
                       "NATOPS", "HandMovementDirection"]   # 锁位2 + 备选前2

    def test_failed_lock_frees_slot_for_backup(self):
        stats = {"SpokenArabicDigits": _st(0.30),           # 低于 40% 落选
                 "EthanolConcentration": _st(0.45),
                 "NATOPS": _st(0.55), "HandMovementDirection": _st(0.50),
                 "Cricket": _st(0.48)}
        got = sp.promotion_verdict(stats)
        assert got == ["EthanolConcentration", "NATOPS",
                       "HandMovementDirection", "Cricket"]

    def test_ceiling_and_deadflat_exclude(self):
        stats = {"SpokenArabicDigits": _st(0.50),
                 "EthanolConcentration": _st(0.45),
                 "NATOPS": _st(0.60, max_r=0.995),          # 贴顶
                 "HandMovementDirection": _st(0.55, rng=0.05),  # 死平
                 "Cricket": _st(0.41)}
        got = sp.promotion_verdict(stats)
        assert got == ["SpokenArabicDigits", "EthanolConcentration", "Cricket"]

    def test_boundary_inclusive_and_empty(self):
        assert sp.promotion_verdict({"Cricket": _st(0.40)}) == ["Cricket"]
        assert sp.promotion_verdict({"Cricket": _st(0.399)}) == []


# ---------------------------------------------------------------- EEG 低权判
class TestEegVerdict:
    def test_pass_at_boundary(self):
        assert sp.eeg_verdict(_st(0.30)) is True

    def test_below_threshold_or_deadflat_fails(self):
        assert sp.eeg_verdict(_st(0.29)) is False
        assert sp.eeg_verdict(_st(0.50, rng=0.05)) is False


# ---------------------------------------------------------------- 靶池三档判
class TestTargetVerdict:
    def test_regular(self):
        m = {"evalgap_med": 0.12, "max_r": 0.95, "headroom_med": 0.08}
        assert sp.target_verdict(m) == "regular"

    def test_showcase_when_no_headroom(self):
        m = {"evalgap_med": 0.12, "max_r": 0.95, "headroom_med": 0.004}
        assert sp.target_verdict(m) == "showcase"

    def test_out_when_not_evaluable(self):
        assert sp.target_verdict(
            {"evalgap_med": 0.03, "max_r": 0.95, "headroom_med": 0.2}) == "out"
        assert sp.target_verdict(
            {"evalgap_med": 0.12, "max_r": 0.995, "headroom_med": 0.2}) == "out"

    def test_boundary_inclusive(self):
        m = {"evalgap_med": 0.05, "max_r": 0.95, "headroom_med": 0.05}
        assert sp.target_verdict(m) == "regular"

    def test_negative_headroom_is_showcase(self):
        # GGS 目录外净空可为负（协议 §4 声明）——仍属"可评不过净空"档
        m = {"evalgap_med": 0.12, "max_r": 0.95, "headroom_med": -0.03}
        assert sp.target_verdict(m) == "showcase"


# ---------------------------------------------------------------- GGS 任务数
class TestGgsTaskCount:
    def test_min_of_eight_and_planned(self):
        assert sp.ggs_task_count(16) == 8
        assert sp.ggs_task_count(8) == 8
        assert sp.ggs_task_count(6) == 6
        assert sp.ggs_task_count(3) == 3
