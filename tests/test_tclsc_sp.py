"""SP1 检验（剪空间 S⁺，results/tclsc_sp/PROTOCOL.md）——先于实现。覆盖：z-bad / strict 标签与可评切片规则、代理确定性、τ 选择（训练折纯度 ≤ 10% 的最小 τ）、
S⁺ 采样器（合法、ẑ ≥ τ、确定性）、空间占比估计、门与读数码。零 GPU、不读真实缓存。"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from autocls.space.cls_space import CLS, CLSSpace


def _g78():
    return importlib.import_module("scripts.78_tclsc_sp")


@pytest.fixture(scope="module")
def space():
    return CLSSpace.from_yaml()


def _rows(n_slices=3, n_cfg=8, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_slices):
        for j in range(n_cfg):
            out.append({"task_id": f"A_{i:02d}", "dataset": "A", "cls_key": f"k{i}_{j}", "origin": "shared" if j < 4 else "random",
                        "iters": 290, "r_prime": float(rng.uniform(0.2, 0.9))})
    return out


class TestLabels:
    def test_zbad_and_strict(self):
        g = _g78()
        rows = _rows()
        bands = {"A_00": (0.5, 0.05), "A_01": (0.9, 0.05), "A_02": (0.3, 0.02)}      # A_01 基线太高 → 不可评
        lab = g.make_labels(rows, bands, margin=0.02)
        assert len(lab) == len(rows)
        for r in lab:
            assert r["zbad"] == (r["z"] < -1.0)
            if r["task_id"] == "A_01":
                assert r["evaluable"] is False and r["strict"] is None
            else:
                m, b = bands[r["task_id"]]
                assert r["evaluable"] is True and r["strict"] == (r["r_prime"] < m - b)
        z_by = {}
        for r in lab:
            z_by.setdefault(r["task_id"], []).append(r["z"])
        for v in z_by.values():
            assert abs(np.mean(v)) < 1e-9 and abs(np.std(v) - 1) < 1e-9

    def test_evaluable_rule(self):
        g = _g78()
        rows = _rows(n_slices=1)
        best = max(r["r_prime"] for r in rows if r["origin"] == "shared")
        assert g.is_evaluable(rows, "A_00", (best - 0.1, 0.05), margin=0.02) is True
        assert g.is_evaluable(rows, "A_00", (best - 0.01, 0.05), margin=0.02) is False


class TestModelAndTau:
    def test_fit_deterministic_and_predict_shape(self, space):
        g = _g78()
        rng = np.random.default_rng(1)
        cfgs = [space.sample_random(rng) for _ in range(60)]
        X = g.encode(space, cfgs)
        z = np.array([float(x.temperature > 1.0) for x in cfgs]) + rng.normal(0, 0.1, len(cfgs))
        m1, m2 = g.fit_quality(X, z, seed=0), g.fit_quality(X, z, seed=0)
        assert np.allclose(g.predict_quality(m1, X), g.predict_quality(m2, X)) and g.predict_quality(m1, X).shape == (60,)

    def test_choose_tau_purity(self):
        g = _g78()
        zhat = np.linspace(-2, 2, 200)
        zbad = zhat < -0.5                                  # 完美排序：ẑ < −0.5 全是坏的
        tau = g.choose_tau(zhat, zbad, target=0.10)
        keep = zhat >= tau
        assert np.mean(zbad[keep]) <= 0.10
        assert g.choose_tau(zhat, zbad, target=0.10) <= g.choose_tau(zhat, zbad, target=0.01)     # 更严的目标 → 更大的 τ
        assert keep.sum() > 0 and tau <= 0.5 + 1e-6 and tau >= -0.5 - 0.15                       # 最小满足纯度的 τ 落在边界附近


class TestSampler:
    def test_sample_splus_valid_and_deterministic(self, space):
        g = _g78()
        rng = np.random.default_rng(2)
        cfgs = [space.sample_random(rng) for _ in range(80)]
        X = g.encode(space, cfgs)
        z = np.array([1.0 if c.temperature <= 1.0 else -1.0 for c in cfgs])
        model = g.fit_quality(X, z, seed=0)
        tau = 0.0
        a = g.sample_splus(space, model, tau, n=10, rng=np.random.default_rng(3))
        b = g.sample_splus(space, model, tau, n=10, rng=np.random.default_rng(3))
        assert [c.to_dict() for c in a] == [c.to_dict() for c in b] and len(a) == 10
        for c in a:
            assert space.validate(c) and g.predict_quality(model, g.encode(space, [c]))[0] >= tau
        frac = g.space_fraction(space, model, tau, n=500, seed=0)
        assert 0.0 <= frac <= 1.0


class TestGates:
    def test_gates_and_verdict(self):
        g = _g78()
        assert g.gate_s1({"auc": 0.9, "spearman": 0.6}) and not g.gate_s1({"auc": 0.8, "spearman": 0.6}) and not g.gate_s1({"auc": 0.9, "spearman": 0.4})
        assert g.gate_s2({"zbad_rate": 0.05, "strict_rate": 0.08}) and not g.gate_s2({"zbad_rate": 0.12, "strict_rate": 0.08})
        assert g.gate_s2({"zbad_rate": 0.05, "strict_rate": None})                      # 无可评切片 → 只看 z-bad
        assert g.gate_s3({"fraction": 0.3, "recall_good": 0.95}) and not g.gate_s3({"fraction": 0.1, "recall_good": 0.95})
        assert g.verdict_sp({"G-S1": True, "G-S2": True, "G-S3": True}) == "SP-GO"
        assert g.verdict_sp({"G-S1": True, "G-S2": True, "G-S3": False}) == "SP-TIGHT"
        assert g.verdict_sp({"G-S1": False, "G-S2": True, "G-S3": True}) == "SP-DEAD"
        assert g.verdict_sp({"G-S1": True, "G-S2": False, "G-S3": True}) == "SP-DEAD"
