"""GEO-S 检验（results/tclsc_geos/PROTOCOL.md）——先于实现。覆盖：18 家族网格（合法、温度惯例、干净基字段）、格构造（29 × 20、key 唯一、截点、官方 options）、
效用表 / 噪声、SH6-3-3 截点模拟（oracle 顺序 = 最好；反序更差）、门与读数码。零 GPU、不读真实数据。"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from autocls.space.cls_space import CLSSpace


def _g82():
    return importlib.import_module("scripts.82_tclsc_geos")


@pytest.fixture(scope="module")
def space():
    return CLSSpace.from_yaml()


class TestGrid:
    def test_family_grid(self, space):
        g = _g82()
        grid = g.family_grid(space)
        assert len(grid) == 18 and len({k for _f, _c, k in grid}) == 18
        for fam, cls, _k in grid:
            assert space.validate(cls) and (cls.loss_type, cls.sim_func, cls.norm) == fam
            assert cls.temperature == (1.0 if cls.loss_type == "triplet" else 0.1)
            assert cls.crop == 0.5 and cls.resize == 0 and cls.rescale == 0 and cls.jitter == 0 and cls.point_mask == 0 and cls.freq_mask == 0 and cls.emb_jitter == 0 and cls.emb_mask == 0
            assert cls.kernel == 3 and cls.pool == "avg" and cls.temporal is True and cls.cross_scale is True
        fams = {f for f, _c, _k in grid}
        assert ("triplet", "dot", "none") in fams and ("info_nce", "cosine", "l2") in fams

    def test_cells(self, space):
        g = _g82()
        cells = g.build_cells(targets=["RacketSports", "Libras"], seeds_anchor=(0, 1))
        assert len(cells) == 2 * (18 + 2)
        keys = [c["key"] for c in cells]
        assert len(set(keys)) == len(keys)
        for c in cells:
            assert c["protocol"] == "ts2vec_official" and c["stage"] == "s1_catalog" and c["options"]["batch_size"] == 8
            assert c["checkpoints"][-1] == c["iters"] and len(c["checkpoints"]) == 3 and c["family"] in {str(f) for f, _c, _k in g.family_grid(space)}
        assert sorted({c["seed"] for c in cells}) == [0, 1] and sum(c["seed"] == 1 for c in cells) == 4


class TestAnalysis:
    def test_utility_and_sh(self):
        g = _g82()
        rows = []
        for d in ("A", "B"):
            for j, f in enumerate(("f0", "f1", "f2", "f3")):
                base = 0.5 + 0.1 * j if d == "A" else 0.8 - 0.1 * j
                rows.append({"dataset": d, "family": f, "seed": 0, "acc": base, "checkpoint_results": [{"iters": 50, "acc": base - 0.2}, {"iters": 100, "acc": base - 0.1}, {"iters": 200, "acc": base}]})
        U, ds, fams = g.utility_table(rows)
        assert ds == ["A", "B"] and fams == ["f0", "f1", "f2", "f3"]
        assert U[0].tolist() == pytest.approx([-0.3, -0.2, -0.1, 0.0]) and U[1].tolist() == pytest.approx([0.0, -0.1, -0.2, -0.3])
        traj = g.trajectories(rows)
        best = g.sh_sim(traj["A"], order=["f3", "f2", "f1", "f0"], k=(2, 1, 1))
        worst = g.sh_sim(traj["A"], order=["f0", "f1", "f2", "f3"], k=(2, 1, 1))
        assert best == pytest.approx(0.8) and worst == pytest.approx(0.6)

    def test_gates_and_verdict(self):
        g = _g82()
        assert g.gate_s1({"oracle_minus_fixed": 0.02, "frac_over_noise": 0.4}) and not g.gate_s1({"oracle_minus_fixed": 0.01, "frac_over_noise": 0.4})
        assert g.gate_s2({"gain_vs_fixed": 0.006, "p_gain": 0.01}) and not g.gate_s2({"gain_vs_fixed": 0.006, "p_gain": 0.2})
        assert g.gate_s3({"delta": 0.006, "p": 0.01}) and not g.gate_s3({"delta": 0.004, "p": 0.01})
        assert g.verdict_geos({"G-S1": True, "G-S2": True, "G-S3": True}) == "GEOS-GO"
        assert g.verdict_geos({"G-S1": True, "G-S2": False, "G-S3": False}) == "GEOS-HEADROOM-ONLY"
        assert g.verdict_geos({"G-S1": False, "G-S2": True, "G-S3": True}) == "GEOS-FLAT"
        assert g.verdict_geos({"G-S1": True, "G-S2": True, "G-S3": False}) == "GEOS-RANK-ONLY"
