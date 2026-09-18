"""AA1 审计（results/tclsc_aa/PROTOCOL.md）——先于实现。覆盖：ICC(1)（完全一致 → 1、纯噪声 → ≈ 0、置换 p）、孪生配对 Δ、名单规则（always-off / on / 按靶符号 / oracle）、
PD 分量分解（开关分量 / 哪档极差 / 跨集一致性）、判定与读数码。零 GPU、不读真实数据。"""
from __future__ import annotations

import importlib

import numpy as np
import pytest


def _g80():
    return importlib.import_module("scripts.80_aug_audit")


class TestICC:
    def test_icc1_perfect_and_noise(self):
        g = _g80()
        rng = np.random.default_rng(0)
        base = rng.normal(0, 1, size=(20, 1))
        M = np.repeat(base, 6, axis=1)                        # 20 组 × 6 评分者，组内完全一致
        r = g.icc1(M)
        assert r["icc"] == pytest.approx(1.0, abs=1e-9)
        N = rng.normal(0, 1, size=(20, 6))
        r2 = g.icc1(N, n_perm=300, seed=0)
        assert abs(r2["icc"]) < 0.3 and r2["p"] > 0.05
        r3 = g.icc1(M + rng.normal(0, 0.1, M.shape), n_perm=300, seed=0)
        assert r3["icc"] > 0.9 and r3["p"] <= 0.01


class TestTwinsAndRules:
    def test_pair_twins_and_rules(self):
        g = _g80()
        rows = []
        for d in ("A", "B"):
            for c in ("c1", "c2", "c3"):
                base = 0.5 + 0.1 * int(c[1])
                rows.append({"dataset": d, "cls_key": c, "twin_of": None, "acc": base})
                rows.append({"dataset": d, "cls_key": c + "m", "twin_of": c, "acc": base + (0.05 if d == "A" else -0.05)})
        M, ds, cs = g.twin_matrix(rows)
        assert M.shape == (2, 3) and ds == ["A", "B"] and cs == ["c1", "c2", "c3"]
        assert np.allclose(M[0], 0.05) and np.allclose(M[1], -0.05)
        off = {("A", "c1"): 0.6, ("A", "c2"): 0.7, ("A", "c3"): 0.8, ("B", "c1"): 0.6, ("B", "c2"): 0.7, ("B", "c3"): 0.8}
        on = {k: v + (0.05 if k[0] == "A" else -0.05) for k, v in off.items()}
        rules = g.list_rules(off, on)
        assert rules["A"]["always_off"] == 0.8 and rules["A"]["always_on"] == pytest.approx(0.85) and rules["A"]["by_sign"] == pytest.approx(0.85) and rules["A"]["oracle"] == pytest.approx(0.85)
        assert rules["B"]["always_off"] == 0.8 and rules["B"]["always_on"] == pytest.approx(0.75) and rules["B"]["by_sign"] == 0.8 and rules["B"]["oracle"] == 0.8


class TestPD:
    def test_pd_components(self):
        g = _g80()
        levels = [0.0, 0.2, 0.5, 0.8]
        pd_a = {0.0: 0.0, 0.2: 0.5, 0.5: 0.6, 0.8: 0.4}
        pd_b = {0.0: 0.0, 0.2: 0.4, 0.5: 0.6, 0.8: 0.5}
        c = g.pd_components(pd_a, levels)
        assert c["onoff"] == pytest.approx(0.0 - 0.5) and c["which_level_range"] == pytest.approx(0.2)
        agree = g.level_agreement([pd_a, pd_b], levels)
        assert 0.0 < agree <= 1.0
        assert g.level_agreement([pd_a, {k: -v for k, v in pd_a.items()}], levels) < 0


class TestVerdict:
    def test_decisions(self):
        g = _g80()
        assert g.decide_d1({"icc": 0.4, "p": 0.01, "by_sign_minus_off": 0.012, "frac_over": 0.4}) and not g.decide_d1({"icc": 0.2, "p": 0.01, "by_sign_minus_off": 0.012, "frac_over": 0.4})
        assert g.decide_d2({"freq_mask": {"icc": 0.5, "frac_over": 0.4}, "affine": {"icc": 0.1, "frac_over": 0.5}}) and not g.decide_d2({"affine": {"icc": 0.1, "frac_over": 0.5}})
        assert g.decide_d3({"a": {"frac_over_null": 0.6, "agreement": 0.2}, "b": {"frac_over_null": 0.7, "agreement": 0.3}, "c": {"frac_over_null": 0.1, "agreement": 0.9}})
        assert not g.decide_d3({"a": {"frac_over_null": 0.6, "agreement": 0.2}, "b": {"frac_over_null": 0.2, "agreement": 0.3}, "c": {"frac_over_null": 0.1, "agreement": 0.9}})
        assert g.verdict_aa({"D1": True, "D2": False, "D3": True}) == "AA-LEVEL-GO"
        assert g.verdict_aa({"D1": True, "D2": False, "D3": False}) == "AA-SWITCH-ONLY"
        assert g.verdict_aa({"D1": False, "D2": False, "D3": True}) == "AA-DEAD"
