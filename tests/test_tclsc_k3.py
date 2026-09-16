"""K3 检验（标签多种子化）——先于实现。覆盖：噪声格计划（shared-10 × rep {1,2}、零撞键）、k 均值标签（三点齐全才纳入、只留 shared、seed 0 逐位）、
δ_k 规则、考侧 12 切片、信任域目录过滤、门与读数码。零 GPU、不读真实缓存。"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from autocls.labeling import ORIGIN_RANDOM, ORIGIN_SHARED
from scripts import pq_common as pq


def _g75():
    return importlib.import_module("scripts.75_tclsc_k3")


def _main_rows(slices=("A_00", "A_x00"), n_shared=3, n_random=2):
    rows = []
    for s in slices:
        for j in range(n_shared):
            for b in pq.BUDGETS:
                rows.append({"task_id": s, "dataset": pq.dataset_of(s), "cls_key": f"S{j}", "origin": ORIGIN_SHARED, "iters": int(b), "seed": 0,
                             "r_prime": 0.5 + 0.01 * j, "error": None})
        for j in range(n_random):
            for b in pq.BUDGETS:
                rows.append({"task_id": s, "dataset": pq.dataset_of(s), "cls_key": f"R{s}_{j}", "origin": ORIGIN_RANDOM, "iters": int(b), "seed": 0,
                             "r_prime": 0.3, "error": None})
    return rows


def _rep_rows(slices=("A_00", "A_x00"), n_shared=3, reps=(1, 2), drop=()):
    rows = []
    for s in slices:
        for j in range(n_shared):
            for rep in reps:
                for b in pq.BUDGETS:
                    if (s, j, rep) in drop:
                        continue
                    rows.append({"task_id": s, "dataset": pq.dataset_of(s), "cls_key": f"S{j}", "origin": ORIGIN_SHARED, "iters": int(b), "seed": rep,
                                 "r_prime": 0.5 + 0.01 * j + 0.03 * rep, "error": None})
    return rows


class TestCellsAndLabels:
    def test_k3_cells_shared_only_two_reps(self):
        g = _g75()
        cells = g.k3_cells(["Handwriting_00", "Handwriting_x03"], space=pq.CLSSpace.from_yaml())
        assert len(cells) == 2 * 10 * 2
        assert {c.origin for c in cells} == {ORIGIN_SHARED} and {c.rep for c in cells} == {1, 2}
        keys = [(c.slice_id, c.cls_key, c.rep) for c in cells]
        assert len(keys) == len(set(keys))
        assert g.REPS == (1, 2) and g.K == 3

    def test_k_mean_rows(self):
        g = _g75()
        main, rep = _main_rows(), _rep_rows(drop=(("A_x00", 2, 1),))     # A_x00 的 S2 缺 rep1 → 三点不齐 → 剔除
        k3 = g.k_mean_rows(main, rep, k=3)
        assert all(r["origin"] == ORIGIN_SHARED for r in k3)
        got = {(r["task_id"], r["cls_key"], r["iters"]): r["r_prime"] for r in k3}
        assert ("A_x00", "S2", 60) not in got and ("A_x00", "S1", 60) in got
        assert got[("A_00", "S1", 60)] == pytest.approx((0.51 + 0.54 + 0.57) / 3)
        assert all(r["k"] == 3 and r["seed"] == 0 for r in k3)
        k1 = g.k_mean_rows(main, rep, k=1)
        assert {(r["task_id"], r["cls_key"], r["iters"]): r["r_prime"] for r in k1}[("A_00", "S1", 60)] == 0.51    # seed 0 逐位
        assert len(k1) == 2 * 3 * 3 and all(r["origin"] == ORIGIN_SHARED for r in k1)

    def test_delta_k(self):
        g = _g75()
        d1 = {("A", 60): 0.105, ("A", 290): 0.18, ("B", 120): 0.0625}
        d3 = g.delta_k(d1, k=3)
        assert d3[("A", 60)] == pytest.approx(0.065) and d3[("A", 290)] == pytest.approx(0.105) and d3[("B", 120)] == pytest.approx(0.04)
        assert g.delta_k(d1, k=1) == d1


class TestExamAndVerdict:
    def test_exam_slices_and_shared_catalog(self):
        g = _g75()
        ex = g.exam_slices("Handwriting")
        assert ex == [f"Handwriting_{i:02d}" for i in range(6)] + [f"Handwriting_x{i:02d}" for i in range(6)]
        cat = {"A_00": [("shared", None, "s1"), ("random", None, "r1"), ("ggs", None, "g")]}
        assert g.shared_catalog(cat) == {"A_00": [("shared", None, "s1")]}
        assert g.TOP_GOOD_K3 == 5

    def test_gates_and_codes(self):
        g = _g75()
        ok = {"n": 504, "mean_a": 0.030, "mean_b": 0.026, "p": 0.01}
        assert g.gate_k1(ok) and not g.gate_k1({**ok, "p": 0.2}) and not g.gate_k1({**ok, "mean_b": 0.0285})
        assert g.gate_k2({"p": 1e-4, "mean_delta": 0.004}) and not g.gate_k2({"p": 0.01, "mean_delta": 0.004})
        assert g.gate_k3({"n": 504, "mean_a": 0.030, "mean_b": 0.027, "p": 0.02}) and not g.gate_k3({"n": 504, "mean_a": 0.030, "mean_b": 0.029, "p": 0.02})
        assert g.verdict_k3({"G-K1": True, "G-K2": True, "G-K3": False}) == "K3-E-WIN"
        assert g.verdict_k3({"G-K1": False, "G-K2": True, "G-K3": True}) == "K3-LABEL-WIN"
        assert g.verdict_k3({"G-K1": False, "G-K2": False, "G-K3": False}) == "K3-FLAT"
        assert g.verdict_k3({"G-K1": True, "G-K2": False, "G-K3": True}) == "K3-LABEL-WIN"

    def test_prior_ranking_uses_training_slices_only(self):
        g = _g75()
        rows = []
        # A / D：S2 > S1 > S0；B / C：S0 被 +0.2 抬到第一（先验 = within-slice z 均值，z 有界，故用两个数据集抬）
        for s, ds, boost in (("A_00", "A", 0.0), ("D_00", "D", 0.0), ("B_00", "B", 0.2), ("C_00", "C", 0.2)):
            for j in range(3):
                for b in pq.BUDGETS:
                    rows.append({"task_id": s, "dataset": ds, "cls_key": f"S{j}", "origin": ORIGIN_SHARED, "iters": int(b),
                                 "r_prime": 0.5 + 0.01 * j + (boost if j == 0 else 0.0)})
        order = g.prior_order(rows, exclude_dataset="B")          # A + C + D：S2 > S1 > S0
        assert order == ["S2", "S1", "S0"]
        order_a = g.prior_order(rows, exclude_dataset="A")        # B + C + D：S0 第一
        assert order_a[0] == "S0"
