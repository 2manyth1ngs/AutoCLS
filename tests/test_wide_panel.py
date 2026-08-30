"""宽任务面板（scripts/24_wide_panel.py）纯逻辑测试。

协议 results/wide_panel/PROTOCOL.md：期中止损带、闸 2 主池排除 selection、
基线计数/复用键。期望值手工推导。
"""
import importlib

wp = importlib.import_module("scripts.24_wide_panel")


class TestInterimFutility:
    def test_um_gate1_failure_triggers(self):
        assert wp.interim_futility(p_um=0.15, dir_um=1, dir_ct=1) is True

    def test_double_negative_direction_triggers(self):
        assert wp.interim_futility(p_um=0.001, dir_um=-1, dir_ct=-1) is True

    def test_healthy_does_not_trigger(self):
        assert wp.interim_futility(p_um=0.001, dir_um=1, dir_ct=-1) is False
        assert wp.interim_futility(p_um=0.099, dir_um=-1, dir_ct=1) is False

    def test_boundary(self):
        assert wp.interim_futility(p_um=0.10, dir_um=1, dir_ct=1) is True  # ≥0.10 含


class TestPrimaryPool:
    def test_selection_excluded(self):
        pool = wp.primary_pool()
        assert "UniMiB-SHAR" not in pool
        assert set(pool) == {"CharacterTrajectories", "UWaveGestureLibrary",
                             "PhonemeSpectra", "Handwriting"}


class TestBaselineCounts:
    def test_counts_valid_rows_only(self):
        rows = [
            {"task_id": "T_00", "seed": 0, "iters": 0, "r_prime": 0.5,
             "split_seed": 7, "classes": [1, 2]},
            {"task_id": "T_00", "seed": 1, "iters": 0, "r_prime": 0.6,
             "split_seed": 7, "classes": [1, 2]},
            {"task_id": "T_00", "seed": 2, "iters": 0, "r_prime": None,
             "split_seed": 7, "classes": [1, 2]},          # 失败行不计
            {"task_id": "T_00", "seed": 3, "iters": 0, "r_prime": 0.4,
             "split_seed": 8, "classes": [1, 2]},          # 非 canonical 划分不计
            {"task_id": "T_00", "seed": 0, "iters": 290, "r_prime": 0.7,
             "split_seed": 7, "classes": [1, 2]},          # 非 iters=0 不计
        ]
        done = wp.baseline_done_keys(rows, split_of=lambda tid: 7)
        assert done == {("T_00", 0), ("T_00", 1)}
