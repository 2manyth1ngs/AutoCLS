"""MX-2 端到端（scripts/25_mx2_endtoend.py）纯逻辑测试。

协议 results/mx2_endtoend/PROTOCOL.md：四臂定义、random@k 解析期望、
排名确定性、确认工单去重、垫底四分位。期望值手工推导。
"""
import importlib

import numpy as np
import pytest

mx = importlib.import_module("scripts.25_mx2_endtoend")


class TestExpectedMaxOfK:
    def test_k1_is_mean(self):
        vals = [0.1, 0.4, 0.7, 0.2]
        assert mx.expected_max_of_k(vals, 1) == pytest.approx(np.mean(vals))

    def test_k_equals_n_is_max(self):
        vals = [0.1, 0.4, 0.7, 0.2]
        assert mx.expected_max_of_k(vals, 4) == pytest.approx(0.7)

    def test_enumeration_small_case(self):
        # 1..4 取 2 的 6 个组合 max 均值 = (2+3+4+3+4+4)/6 = 10/3
        assert mx.expected_max_of_k([1, 2, 3, 4], 2) == pytest.approx(10 / 3)

    def test_ties_ok(self):
        assert mx.expected_max_of_k([0.5, 0.5, 0.5], 2) == pytest.approx(0.5)


class TestOrderCandidates:
    def test_wins_dominate(self):
        # 3 候选：V 中 2 胜 1 胜 0 胜
        V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        order = mx.order_candidates(V, np.zeros(3), ["a", "b", "c"])
        assert order == [2, 1, 0]

    def test_tiebreak_by_score_then_key(self):
        V = np.zeros((3, 3), dtype=int)          # 全平
        tb = np.array([0.2, 0.8, 0.2])
        order = mx.order_candidates(V, tb, ["z", "m", "a"])
        # 得分 0.8 第一；剩余两个同得分按 key 升序 a < z
        assert order == [1, 2, 0]


class TestTaskArms:
    TRUTH = {"c1": 0.9, "c2": 0.8, "c3": 0.5, "c4": 0.4}

    def test_arms_values(self):
        arms = mx.task_arms(self.TRUTH, ["c2", "c3", "c1", "c4"],
                            ggs_r=0.7, k_top=3)
        assert arms["oracle"] == pytest.approx(0.9)
        assert arms["tclsc1"] == pytest.approx(0.8)       # top-1 = c2
        assert arms["tclsc3"] == pytest.approx(0.9)       # top-3 含 c1
        assert arms["ggs"] == pytest.approx(0.7)
        assert arms["random1"] == pytest.approx(np.mean(list(self.TRUTH.values())))
        assert arms["random3"] == pytest.approx(
            mx.expected_max_of_k(list(self.TRUTH.values()), 3))
        assert arms["bottom"] == pytest.approx(0.4)

    def test_regret_vs_oracle(self):
        arms = mx.task_arms(self.TRUTH, ["c3", "c4", "c2", "c1"],
                            ggs_r=0.95, k_top=3)
        reg = mx.regret_of(arms)
        assert reg["tclsc1"] == pytest.approx(0.4)        # 0.9-0.5
        assert reg["oracle"] == pytest.approx(0.0)
        # GGS 超出目录 oracle → 负 regret 允许（协议 §2 声明）
        assert reg["ggs"] == pytest.approx(-0.05)


class TestConfirmWorklist:
    def test_dedup_and_sorted(self):
        wl = mx.confirm_worklist(
            top1={"t1": "kA", "t2": "kB"},
            oracle={"t1": "kA", "t2": "kC"},
            ggs_key="kG")
        # t1: top1==oracle 去重 → {kA, kG}；t2: {kB, kC, kG}
        assert wl == sorted([("t1", "kA"), ("t1", "kG"),
                             ("t2", "kB"), ("t2", "kC"), ("t2", "kG")])


class TestBottomQuartile:
    def test_hit_and_miss(self):
        truth = {f"c{i}": i / 20 for i in range(20)}      # 0.00..0.95
        # 垫底四分位 = 倒数 5（值 ≤ 0.20 位）
        assert mx.bottom_quartile_hit(truth, "c0") is True
        assert mx.bottom_quartile_hit(truth, "c4") is True
        assert mx.bottom_quartile_hit(truth, "c5") is False
        assert mx.bottom_quartile_hit(truth, "c19") is False


class TestDirections:
    def test_protocol_locked_composition(self):
        assert mx.DIRECTIONS["A"] == {"train": "UniMiB-SHAR",
                                      "test": "Handwriting"}
        assert mx.DIRECTIONS["B"] == {"train": "Handwriting",
                                      "test": "UniMiB-SHAR"}
        assert mx.K_TOP == 3 and mx.SEEDS == (0, 1, 2)
        assert mx.CONFIRM_REPS == (1, 2)
        assert mx.TS2VEC_PAPER_HW == pytest.approx(0.515)
