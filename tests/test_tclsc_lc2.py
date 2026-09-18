"""LC2 检验（未见配置学习曲线，results/tclsc_lc2/PROTOCOL.md）——先于实现。覆盖：曲线点的训练行构造（R0 / R3 / R6 / RA / RN；三截点同进同出、固定种子、S⁺ 过滤）、
考卷构造（30 条：shared + random-10 + NB 10，未见配置与训练行零交集）、未见配置指标（配对 acc / 非坏区配对 / regret3 / over / above_trust）、门与读数码。零 GPU、不读真实缓存。"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from autocls.labeling import ORIGIN_RANDOM, ORIGIN_SHARED


def _g79():
    return importlib.import_module("scripts.79_tclsc_lc2")


def _rows(slice_ids=("A_00", "A_01", "B_00"), n_rand=10, n_nb=10):
    out = []
    for s in slice_ids:
        d = s.split("_")[0]
        for b in (60, 120, 290):
            for j in range(10):
                out.append({"task_id": s, "dataset": d, "cls_key": f"S{j}", "origin": ORIGIN_SHARED, "iters": b, "r_prime": 0.5 + 0.02 * j, "cls": {"j": j}, "k": 3})
            for j in range(n_rand):
                out.append({"task_id": s, "dataset": d, "cls_key": f"{s}_R{j}", "origin": ORIGIN_RANDOM, "iters": b, "r_prime": 0.3 + 0.04 * j, "cls": {"j": j}, "k": 1, "src": "random"})
            for j in range(n_nb):
                out.append({"task_id": s, "dataset": d, "cls_key": f"{s}_N{j}", "origin": ORIGIN_RANDOM, "iters": b, "r_prime": 0.4 + 0.03 * j, "cls": {"j": j}, "k": 1, "src": "nb"})
    return out


class TestCurvePoints:
    def test_points_structure_and_determinism(self):
        g = _g79()
        rows = _rows()
        good = {r["cls_key"] for r in rows if r.get("src") == "random" and int(r["cls_key"][-1]) >= 3}    # 模拟 S⁺ 过滤：R3..R9 好、R0..R2 坏
        pts = g.curve_rows(rows, splus_keys=good, seed=0)
        assert list(pts) == ["R0", "R3", "R6", "RA", "RN"]
        for name, sub in pts.items():
            keys_per_slice = {}
            for r in sub:
                keys_per_slice.setdefault(r["task_id"], set()).add(r["cls_key"])
            for s, ks in keys_per_slice.items():
                n_sh = sum(k.startswith("S") for k in ks); n_r = sum("_R" in k for k in ks); n_n = sum("_N" in k for k in ks)
                assert n_sh == 10
                if name == "R0":
                    assert n_r == 0 and n_n == 0
                elif name == "R3":
                    assert n_r == 3 and n_n == 0
                elif name == "R6":
                    assert n_r == 6 and n_n == 0
                elif name == "RA":
                    assert n_r == 7 and n_n == 0                    # 全部 S⁺ 随机 = 7
                else:
                    assert n_r == 7 and n_n == 10
                for k in ks:
                    if "_R" in k:
                        assert k in good                                # 只取 S⁺ 内
                        assert sorted(r["iters"] for r in sub if r["task_id"] == s and r["cls_key"] == k) == [60, 120, 290]   # 三截点同进
        assert [r["cls_key"] for r in pts["R3"]] == [r["cls_key"] for r in g.curve_rows(rows, splus_keys=good, seed=0)["R3"]]
        assert set(r["cls_key"] for r in pts["R3"]) <= set(r["cls_key"] for r in pts["R6"]) <= set(r["cls_key"] for r in pts["RA"])   # 嵌套


class TestExam:
    def test_exam_catalog_unseen_disjoint(self):
        g = _g79()
        rows = _rows()
        cat = g.exam_catalog(rows, "A_00")
        assert len(cat) == 30 and sum(o == ORIGIN_SHARED for o, _c, _k in cat) == 10
        unseen = g.unseen_keys(cat)
        assert len(unseen) == 20 and all(k.startswith("A_00_") for k in unseen)
        train = [r for r in rows if r["dataset"] != "A"]
        assert not (unseen & {r["cls_key"] for r in train})

    def test_unseen_metrics(self):
        g = _g79()
        truth = {f"S{j}": 0.5 + 0.02 * j for j in range(10)}
        truth.update({f"U{j}": 0.3 + 0.05 * j for j in range(20)})          # U19 = 1.25 最好
        z = {k: (v - np.mean(list(truth.values()))) / np.std(list(truth.values())) for k, v in truth.items()}
        unseen = {k for k in truth if k.startswith("U")}
        perfect = sorted(truth, key=lambda k: -truth[k])
        m = g.unseen_metrics(perfect, truth, z, unseen, delta=0.01)
        assert m["unseen_pair_acc"] == 1.0 and m["unseen_good_pair_acc"] == 1.0 and m["unseen_regret3"] == 0.0 and m["over"] == 0.0
        reverse = perfect[::-1]
        m2 = g.unseen_metrics(reverse, truth, z, unseen, delta=0.01)
        assert m2["unseen_pair_acc"] == 0.0 and m2["unseen_regret3"] > 0
        assert m["above_trust_pred"] == m["above_trust_true"]


class TestGates:
    def test_gates_and_verdict(self):
        g = _g79()
        assert g.gate_l1({"R3_RA": {"delta": 0.025, "p": 0.01}, "RA_RN": {"delta": 0.012, "p": 0.3}})
        assert not g.gate_l1({"R3_RA": {"delta": 0.025, "p": 0.2}, "RA_RN": {"delta": 0.012, "p": 0.3}})
        assert not g.gate_l1({"R3_RA": {"delta": 0.025, "p": 0.01}, "RA_RN": {"delta": 0.005, "p": 0.3}})
        assert g.gate_l2({"R3": 0.01, "RN": 0.025}) and not g.gate_l2({"R3": 0.02, "RN": 0.025})
        assert g.gate_l3({"R3": 0.30, "RN": 0.15}) and not g.gate_l3({"R3": 0.30, "RN": 0.25})
        assert g.verdict_lc2({"G-L1": True, "G-L2": False, "G-L3": False}) == "LC2-RISING"
        assert g.verdict_lc2({"G-L1": False, "G-L2": True, "G-L3": True}) == "LC2-CALIBRATE-ONLY"
        assert g.verdict_lc2({"G-L1": False, "G-L2": True, "G-L3": False}) == "LC2-FLAT"
