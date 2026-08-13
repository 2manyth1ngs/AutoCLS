"""8_forecast_noise_probe 分析函数单测（判据实现先钉死再跑数据，铁律 5）。

判据（results/forecast_noise_probe/PROTOCOL.md 预注册）：
- 主判据 = 每 iters 档"同 (cls) 跨 3 weight_seed 的相对重复差 |ΔMSE|/cell均值"
  的 P95（分位公式与 2c calibrate 逐字同式），三档取最低者入判：
  <5% 绿灯 / 5–15% 中间档 / >15% 红灯（边界值归中间档）。
- CV（ddof=1）为次要口径，随表报告。
"""
import importlib

import numpy as np
import pytest

probe = importlib.import_module("scripts.8_forecast_noise_probe")


class TestSeeds:
    def test_rep0_is_canonical_and_deterministic(self):
        from autocls.labeling import run_seed_for

        s = probe.seeds_for("abc123", 0)
        assert s == run_seed_for(probe.TASK_ID, "abc123")
        assert probe.seeds_for("abc123", 0) == s  # 幂等

    def test_noise_reps_differ_from_canonical_and_each_other(self):
        seeds = {probe.seeds_for("abc123", rep) for rep in (0, 1, 2)}
        assert len(seeds) == 3

    def test_different_cls_different_seed(self):
        assert probe.seeds_for("abc123", 0) != probe.seeds_for("def456", 0)


class TestPlanWork:
    def test_full_grid_unique_keys(self):
        cls_keys = [f"k{i}" for i in range(10)]
        work = probe.plan_work(cls_keys)
        assert len(work) == 10 * 3 * len(probe.ITERS_LEVELS)
        assert len({(it, k, rep) for it, k, rep in work}) == len(work)

    def test_cheapest_iters_level_first(self):
        work = probe.plan_work(["a"])
        assert [it for it, _, _ in work[:3]] == [min(probe.ITERS_LEVELS)] * 3


class TestLoadCells:
    def test_groups_by_iters_cls_and_later_row_wins(self):
        rows = [
            {"iters": 30, "cls_key": "a", "seed": 0, "r_prime": 0.5},
            {"iters": 30, "cls_key": "a", "seed": 1, "r_prime": 0.6},
            {"iters": 30, "cls_key": "a", "seed": 0, "r_prime": 0.55},  # 重放后行覆盖
            {"iters": 60, "cls_key": "a", "seed": 0, "r_prime": 0.4},
            {"iters": 30, "cls_key": "b", "seed": 0, "r_prime": None},  # 失败条不进 cell
        ]
        cells = probe.load_cells(rows)
        assert cells[(30, "a")] == {0: 0.55, 1: 0.6}
        assert cells[(60, "a")] == {0: 0.4}
        assert (30, "b") not in cells


class TestNoiseStats:
    def _cells(self):
        # 两个 cell：a 的 3 rep = 0.50/0.55/0.60（均值 0.55），b = 1.0/1.0/1.1
        return {
            (30, "a"): {0: 0.50, 1: 0.55, 2: 0.60},
            (30, "b"): {0: 1.00, 1: 1.00, 2: 1.10},
            (60, "a"): {0: 0.50, 1: 0.50, 2: 0.50},
        }

    def test_pairwise_abs_and_relative_diffs(self):
        st = probe.noise_stats(probe_cells=self._cells(), iters=30)
        # a: |Δ| = .05/.10/.05；b: 0/.1/.1 → abs 共 6 个
        assert st["n_pairs"] == 6
        np.testing.assert_allclose(st["abs"]["max"], 0.10)
        # rel：a 以 cell 均值 0.55 归一 → .0909/.1818/.0909；b 以 31/30 归一
        np.testing.assert_allclose(st["rel"]["max"], 0.10 / 0.55, rtol=1e-6)

    def test_p95_formula_matches_2c_semantics(self):
        # sorted 后取 round(0.95*(n-1)) 索引（与 2c calibrate/7_swa 逐字同式）
        diffs = [0.01 * i for i in range(20)]
        assert probe.pctl(diffs, 0.95) == diffs[int(round(0.95 * 19))]

    def test_cv_uses_ddof1(self):
        st = probe.noise_stats(probe_cells=self._cells(), iters=30)
        cv_a = np.std([0.50, 0.55, 0.60], ddof=1) / np.mean([0.50, 0.55, 0.60])
        assert any(abs(c["cv"] - cv_a) < 1e-9 for c in st["cells"])

    def test_zero_noise_cell(self):
        st = probe.noise_stats(probe_cells=self._cells(), iters=60)
        assert st["rel"]["max"] == 0.0 and st["abs"]["max"] == 0.0

    def test_incomplete_cell_excluded(self):
        cells = {(30, "a"): {0: 0.5}, (30, "b"): {0: 1.0, 1: 1.2, 2: 1.1}}
        st = probe.noise_stats(probe_cells=cells, iters=30)
        assert st["n_cells"] == 1  # 只剩 b；单 rep 的 a 无重复差可算


class TestVerdict:
    def test_green_mid_red_mapping(self):
        assert probe.verdict_of(0.049)["verdict"] == "green"
        assert probe.verdict_of(0.050)["verdict"] == "mid"  # 边界归中间档
        assert probe.verdict_of(0.140)["verdict"] == "mid"
        assert probe.verdict_of(0.150)["verdict"] == "mid"  # 边界归中间档
        assert probe.verdict_of(0.151)["verdict"] == "red"

    def test_best_level_selection(self):
        rel = {30: 0.20, 60: 0.04, 290: 0.08}
        v = probe.overall_verdict(rel)
        assert v["best_iters"] == 60
        assert v["verdict"] == "green"
