"""Step 1 encoder 探针（scripts/13_step1_encoder_probe.py）纯逻辑测试。

期望值手工推导：零假设带、越上带计数与 best gain、信号问共用门、判读表。
"""
import importlib

s1 = importlib.import_module("scripts.13_step1_encoder_probe")


class TestNullBands:
    def test_band_is_2x_max_sigma_quantum(self):
        # 值 [0.5, 0.5, 0.5] → σ=0 → band=2×量子
        out = s1.null_bands({"t": {"values": [0.5, 0.5, 0.5], "quantum": 0.01}})
        assert out["t"] == (0.5, 0.02)

    def test_sigma_dominates(self):
        # [0.4, 0.6] → mean 0.5、σ≈0.1414 → band≈0.2828 > 2×量子
        (mean, band), = s1.null_bands(
            {"t": {"values": [0.4, 0.6], "quantum": 0.001}}).values()
        assert abs(mean - 0.5) < 1e-12 and abs(band - 0.28284271) < 1e-6


class TestCountAboveBand:
    def test_hand_computed(self):
        bands = {"t1": (0.5, 0.05)}
        cells = {("t1", "a"): 0.60,   # 越上带（>0.55）
                 ("t1", "b"): 0.54,   # 带内
                 ("t1", "c"): 0.40}   # 带下
        out = s1.count_above_band(cells, bands)
        assert out["n_above"] == 1 and out["n_total"] == 3
        assert abs(out["best_gain"] - 0.10) < 1e-12

    def test_unknown_task_skipped(self):
        out = s1.count_above_band({("tX", "a"): 0.9}, {"t1": (0.5, 0.05)})
        assert out["n_total"] == 0 and out["best_gain"] is None


class TestQSignal:
    def test_pass_on_one_arm(self):
        per_arm = {60: {"n_above": 3, "n_total": 20, "best_gain": 0.02},
                   240: {"n_above": 7, "n_total": 20, "best_gain": 0.08}}
        out = s1.q_signal(per_arm, n_above_min=6, best_gain_min=0.05)
        assert out["passed"] and out["ok_arms"] == [240]

    def test_both_conditions_required(self):
        per_arm = {60: {"n_above": 9, "n_total": 20, "best_gain": 0.03}}
        assert not s1.q_signal(per_arm, n_above_min=6, best_gain_min=0.05)["passed"]

    def test_none_best_gain(self):
        per_arm = {60: {"n_above": 9, "n_total": 20, "best_gain": None}}
        assert not s1.q_signal(per_arm, n_above_min=6, best_gain_min=0.0)["passed"]


class TestVerdictCell:
    def test_table(self):
        assert s1.verdict_cell(True, True, False, True) == "transformer_closed"
        assert s1.verdict_cell(True, True, True, False) == "transformer_closed"
        assert s1.verdict_cell(True, False, True, True) == "GO_full"
        assert s1.verdict_cell(False, True, True, True) == "GO_full"
        assert s1.verdict_cell(False, False, True, True) == "GO_narrow"
