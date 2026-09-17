"""NB1 检验（源侧邻域标签 + 开放空间排序）——先于实现。覆盖：局部变异（汉明恰为 h、比例档 ±1/±2、约束后合法、去重）、固定邻居 / 切片邻居的确定性与结构、
切片中心（k3 前 2）、目录（20 条 = shared 10 + 固定 4 + 切片 6）、网格零撞键、标签合并（shared k3 + 邻居 k1、真值 k1）、净空统计、高估签名、门与读数码。零 GPU、不读真实缓存。"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from autocls.labeling import ORIGIN_RANDOM, ORIGIN_SHARED, cls_key
from autocls.space.cls_space import CLS, CLSSpace
from scripts import pq_common as pq


def _g76():
    return importlib.import_module("scripts.76_tclsc_nb")


@pytest.fixture(scope="module")
def space():
    return CLSSpace.from_yaml()


@pytest.fixture(scope="module")
def shared(space):
    return pq.load_probe_cls(pq.SHARED_CATALOG, pq.N_SHARED)[0]        # [(CLS, key)]


def _hamming(a: CLS, b: CLS, fields) -> int:
    return sum(1 for f in fields if getattr(a, f) != getattr(b, f))


class TestMutation:
    def test_mutate_exact_hamming_local_ratio_valid(self, space, shared):
        g = _g76()
        base = shared[0][0]
        rng = np.random.default_rng(0)
        for h in (1, 2, 3):
            for _ in range(20):
                m = g.mutate(base, h, rng, space)
                assert space.validate(m)
                assert _hamming(base, m, space.field_names) == h
                for f in space.field_names:
                    a, b = getattr(base, f), getattr(m, f)
                    if a != b and g.is_ratio_field(space, f):
                        ch = space.choices(f)
                        assert abs(ch.index(a) - ch.index(b)) <= 2           # 比例档只走 ±1 / ±2
        assert g.mutate(base, 1, np.random.default_rng(3), space) == g.mutate(base, 1, np.random.default_rng(3), space)   # 确定性

    def test_fixed_and_slice_neighbors(self, space, shared):
        g = _g76()
        centers = [c for c, _k in shared[:2]]
        fx = g.fixed_neighbors(space, centers, seed=7)
        assert len(fx) == 4 and len({cls_key(c) for _ctr, c, _h in fx}) == 4
        assert sorted(h for _c, _m, h in fx) == [1, 1, 2, 2]
        assert all(cls_key(m) not in {k for _c, k in shared} for _c, m, _h in fx)
        assert fx == g.fixed_neighbors(space, centers, seed=7)
        taken = {k for _c, k in shared} | {cls_key(m) for _c, m, _h in fx}
        sl = g.slice_neighbors(space, "Handwriting_x03", centers, taken=set(taken))
        assert len(sl) == 6 and sorted(h for _c, _m, h in sl) == [1, 1, 2, 2, 3, 3]
        keys = [cls_key(m) for _c, m, _h in sl]
        assert len(set(keys)) == 6 and not (set(keys) & taken)
        assert sl == g.slice_neighbors(space, "Handwriting_x03", centers, taken=set(taken))         # 同 id 同结果
        assert [cls_key(m) for _c, m, _h in g.slice_neighbors(space, "Handwriting_x04", centers, taken=set(taken))] != keys   # 换 id 换结果
        for ctr, m, h in sl:
            assert _hamming(ctr, m, space.field_names) == h and space.validate(m)


class TestCatalogAndLabels:
    def test_slice_centers_from_k3(self):
        g = _g76()
        rows = []
        for b in pq.BUDGETS:
            for j, v in enumerate((0.5, 0.7, 0.6, 0.4)):
                rows.append({"task_id": "A_00", "dataset": "A", "cls_key": f"S{j}", "origin": ORIGIN_SHARED, "iters": int(b), "r_prime": v})
        assert g.slice_centers(rows, "A_00", n=2) == ["S1", "S2"]

    def test_catalog_structure_and_plan(self, space, shared):
        g = _g76()
        centers = {c_k: c for c, c_k in shared}
        cat = g.build_catalog(space, shared, ["A_00", "B_x01"], slice_center_keys={"A_00": [shared[0][1], shared[1][1]], "B_x01": [shared[2][1], shared[0][1]]}, seed=7)
        assert set(cat) == {"shared", "fixed", "random", "centers"}
        assert len(cat["shared"]) == 10 and len(cat["fixed"]) == 4 and set(cat["random"]) == {"A_00", "B_x01"} and all(len(v) == 6 for v in cat["random"].values())
        entries = g.catalog_entries(cat, "A_00")
        assert len(entries) == 20 and sum(1 for o, _c, _k in entries if o == ORIGIN_SHARED) == 14 and sum(1 for o, _c, _k in entries if o == ORIGIN_RANDOM) == 6
        cells = g.nb_cells(cat, ["A_00", "B_x01"])
        assert len(cells) == 20 and len({(c.slice_id, c.cls_key) for c in cells}) == 20         # 固定 4 + 切片 6，shared 不重跑
        assert {c.origin for c in cells} == {ORIGIN_SHARED, ORIGIN_RANDOM}

    def test_merge_labels_k1_truth(self):
        g = _g76()
        k3 = [{"task_id": "A_00", "dataset": "A", "cls_key": "S0", "origin": ORIGIN_SHARED, "iters": 60, "r_prime": 0.55, "k": 3}]
        k1 = [{"task_id": "A_00", "dataset": "A", "cls_key": "S0", "origin": ORIGIN_SHARED, "iters": 60, "r_prime": 0.5, "k": 1}]
        nb = [{"task_id": "A_00", "dataset": "A", "cls_key": "N0", "origin": ORIGIN_RANDOM, "iters": 60, "r_prime": 0.6, "error": None, "nb_center": "S0", "nb_h": 1}]
        train = g.merge_labels(k3, nb)
        assert {r["cls_key"]: r["r_prime"] for r in train} == {"S0": 0.55, "N0": 0.6}
        truth = g.merge_labels(k1, nb)
        assert {r["cls_key"]: r["r_prime"] for r in truth} == {"S0": 0.5, "N0": 0.6}

    def test_headroom_and_overestimation(self):
        g = _g76()
        rows = []
        for s, best_s, best_n in (("A_00", 0.60, 0.65), ("A_01", 0.60, 0.58), ("B_00", 0.70, 0.80)):
            rows += [{"task_id": s, "dataset": s[0], "cls_key": "S0", "origin": ORIGIN_SHARED, "iters": 290, "r_prime": best_s},
                     {"task_id": s, "dataset": s[0], "cls_key": "N0", "origin": ORIGIN_RANDOM, "iters": 290, "r_prime": best_n, "nb_center": "S0", "nb_h": 1}]
        hd = g.headroom(rows, delta={("A", 290): 0.06, ("B", 290): 0.06}, budget=290)
        assert hd["n"] == 3 and hd["frac_positive"] == pytest.approx(2 / 3) and hd["frac_over_delta"] == pytest.approx(1 / 3)
        assert hd["by_dataset"]["B"]["median_gap"] == pytest.approx(0.10)
        assert g.gate_n1(hd) and not g.gate_n1({**hd, "frac_positive": 0.3})
        assert g.gate_n3({"over": 0.10, "cross_p": 1e-4}) and not g.gate_n3({"over": 0.20, "cross_p": 1e-4}) and not g.gate_n3({"over": 0.10, "cross_p": 0.01})
        assert g.verdict_nb({"G-N1": False, "G-N2": True, "G-N3": True}) == "NB-NO-HEADROOM"
        assert g.verdict_nb({"G-N1": True, "G-N2": True, "G-N3": True}) == "NB-OPEN-WIN"
        assert g.verdict_nb({"G-N1": True, "G-N2": True, "G-N3": False}) == "NB-OPEN-OVER"
        assert g.verdict_nb({"G-N1": True, "G-N2": False, "G-N3": True}) == "NB-OPEN-FLAT"
