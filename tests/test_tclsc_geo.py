"""GEO-E 检验（results/tclsc_geo/PROTOCOL.md）——先于实现。覆盖：家族映射、家族效用表 u_d(f)、LODO 诚实的固定最好家族、岭预测器（确定性、argmax、可分离场景命中 1.0）、置换 p、门与读数码。零 GPU、不读真实数据。"""
from __future__ import annotations

import importlib

import numpy as np
import pytest


def _g81():
    return importlib.import_module("scripts.81_tclsc_geo")


class TestFamilies:
    def test_family_of_and_table(self):
        g = _g81()
        assert g.family_of({"loss_type": "triplet", "sim_func": "dot", "norm": "none", "temperature": 10.0}) == ("triplet", "dot", "none")
        rows = [{"dataset": "A", "cls_key": "k1", "acc": 0.8, "cls": {"loss_type": "triplet", "sim_func": "dot", "norm": "none"}},
                {"dataset": "A", "cls_key": "k2", "acc": 0.7, "cls": {"loss_type": "triplet", "sim_func": "dot", "norm": "none"}},
                {"dataset": "A", "cls_key": "k3", "acc": 0.9, "cls": {"loss_type": "info_nce", "sim_func": "euclidean", "norm": "none"}},
                {"dataset": "B", "cls_key": "k1", "acc": 0.6, "cls": {"loss_type": "triplet", "sim_func": "dot", "norm": "none"}},
                {"dataset": "B", "cls_key": "k3", "acc": 0.5, "cls": {"loss_type": "info_nce", "sim_func": "euclidean", "norm": "none"}}]
        U, ds, fams = g.utility_table(rows)
        assert ds == ["A", "B"] and fams == [("info_nce", "euclidean", "none"), ("triplet", "dot", "none")]
        assert U[0].tolist() == pytest.approx([0.0, -0.1]) and U[1].tolist() == pytest.approx([-0.1, 0.0])
        assert g.champion(U, fams) == [("info_nce", "euclidean", "none"), ("triplet", "dot", "none")]

    def test_fixed_best_family_on_training_fold(self):
        g = _g81()
        U = np.array([[0.0, -0.1], [-0.1, 0.0], [-0.05, 0.0], [-0.2, 0.0]])
        assert g.fixed_best(U, train_idx=[1, 2, 3]) == 1
        assert g.fixed_best(U, train_idx=[0, 1]) in (0, 1)


class TestPredictor:
    def test_ridge_lodo_separable(self):
        g = _g81()
        rng = np.random.default_rng(0)
        n = 30
        X = rng.normal(0, 1, size=(n, 5))
        U = np.zeros((n, 2))
        U[:, 0] = np.where(X[:, 0] > 0, 0.0, -0.1)          # 家族 0 在 x0>0 时更好
        U[:, 1] = np.where(X[:, 0] > 0, -0.1, 0.0)
        res = g.lodo_predict(X, U, alphas=(0.1, 1.0, 10.0), seed=0)
        assert res["picks"].shape == (n,) and res["hit_rate"] >= 0.9
        res2 = g.lodo_predict(X, U, alphas=(0.1, 1.0, 10.0), seed=0)
        assert np.array_equal(res["picks"], res2["picks"])
        ev = g.evaluate_picks(U, res["picks"], fixed_by_fold=res["fixed_by_fold"])
        assert ev["gain_vs_fixed"] > 0.03 and ev["mean_regret"] > -0.02

    def test_permutation_p(self):
        g = _g81()
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, size=(24, 4)); U = rng.normal(0, 0.05, size=(24, 3)); U -= U.max(axis=1, keepdims=True)
        p = g.permutation_p(X, U, stat="gain", n_perm=50, seed=0)
        assert 0.0 < p["p"] <= 1.0 and "observed" in p


class TestGates:
    def test_gates_and_verdict(self):
        g = _g81()
        assert g.gate_g1({"gain_vs_fixed": 0.006, "p_gain": 0.02}) and not g.gate_g1({"gain_vs_fixed": 0.004, "p_gain": 0.02}) and not g.gate_g1({"gain_vs_fixed": 0.01, "p_gain": 0.2})
        assert g.gate_g2({"hit_rate": 0.55, "majority": 0.4, "p_hit": 0.01}) and not g.gate_g2({"hit_rate": 0.45, "majority": 0.4, "p_hit": 0.01})
        assert g.verdict_geo({"G-G1": True, "G-G2": False}) == "GEO-E-GO"
        assert g.verdict_geo({"G-G1": False, "G-G2": True}) == "GEO-E-WEAK"
        assert g.verdict_geo({"G-G1": False, "G-G2": False}) == "GEO-E-DEAD"
