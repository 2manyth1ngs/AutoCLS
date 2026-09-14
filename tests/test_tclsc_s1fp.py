"""WP16-TCLSC 阶段 1 转导指纹臂——先于实现（铁律 5）。协议 results/tclsc_s1fp/PROTOCOL.md。

A. `scripts/33`：`FP_ARMS`；`ctx_kind="fp_budget"` 维度 11；`fp_probes`（训练折先验前 K 的 shared）；`fingerprint`（形状 6 ⊕ 水平 2）；
   `fp_standardizer`；`context_for` 的 fp 向量；冒烟目录训 `fp_diff_hard` → meta 记 fp_probes / fp_mu / fp_sd / fp_budget，cross 出块。
B. `scripts/59_tclsc_s1fp.py`：ctx 行（normal / cross / zero）、nonprobe 视图过滤、`probe6_top3` 参照、门与读数码。
零 GPU、不读真实缓存。
"""
from __future__ import annotations

import importlib
import json

import numpy as np
import pytest
import torch

from scripts import pq_common as pq

m33 = importlib.import_module("scripts.33_pq_comparator_loo")
g34 = importlib.import_module("scripts.34_pq_search")

K = 6
FP_DIM = K + 2
CTX_DIM = FP_DIM + len(pq.BUDGETS)      # 11
FP_NAMES = ("fp_concat_hard", "fp_diff_hard", "fp_diff_zdiff")


def _g59():
    return importlib.import_module("scripts.59_tclsc_s1fp")


def _cache(slices, keys, seed=0, budgets=pq.BUDGETS):
    """合成 R' 缓存 {(slice, key, iters): rec}：前 K 个 key 标 shared，其余 random。"""
    rng = np.random.default_rng(seed)
    out = {}
    for s in slices:
        for j, k in enumerate(keys):
            base = 0.9 - 0.03 * j
            for b in budgets:
                out[(s, k, int(b))] = {"task_id": s, "cls_key": k, "iters": int(b), "r_prime": float(base + 0.02 * rng.standard_normal()),
                                       "origin": "shared" if j < K + 2 else "random", "cls": {}}
    return out


# ================================================================ A. 33 号
class TestFpArms:
    def test_table_shape(self):
        assert set(m33.FP_ARMS) == set(FP_NAMES)
        assert not (set(m33.FP_ARMS) & (set(m33.ARMS) | set(m33.L3_ARMS) | set(m33.PHI_ARMS)))
        for name, spec in m33.FP_ARMS.items():
            assert spec.split == "S1" and spec.seeds == (0, 1, 2) and spec.iron_law_6_compliant is True
            assert spec.task_branch == "ctx" and spec.ctx_kind == "fp_budget"
            assert spec.head == name.split("_")[1] and spec.label_mode == name.split("_")[2]
        assert m33.arm_spec("fp_diff_hard") is m33.FP_ARMS["fp_diff_hard"]
        assert m33.FP_K == K and m33.FP_BUDGET == 60

    def test_ctx_dim(self):
        assert m33.ctx_dim_for(m33.FP_ARMS["fp_diff_hard"], pq.BUDGETS, pq.PQ_DATASETS) == CTX_DIM

    def test_fp_probes_prior_top_k_shared_only(self):
        keys = [f"k{j}" for j in range(12)]
        cache = _cache(["Cricket_00", "Cricket_01", "UWaveGestureLibrary_00"], keys)
        probes = m33.fp_probes(cache, ["Cricket_00", "Cricket_01", "UWaveGestureLibrary_00"], k=K)
        assert len(probes) == K and all(p in keys[:K + 2] for p in probes)      # 只从 shared 里选
        # 合成缓存里 r 随 j 递减 → 先验前 6 应是 k0..k5
        assert probes == [f"k{j}" for j in range(K)]

    def test_fp_probes_ignores_test_slices(self):
        keys = [f"k{j}" for j in range(12)]
        cache = _cache(["Cricket_00", "Handwriting_00"], keys)
        # 把考侧切片的 k7 抬到最高：若被算进先验会改变选择
        for b in pq.BUDGETS:
            cache[("Handwriting_00", "k7", int(b))]["r_prime"] = 5.0
        assert "k7" not in m33.fp_probes(cache, ["Cricket_00"], k=K)

    def test_fingerprint_shape_and_level(self):
        m = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        f = m33.fingerprint(m)
        assert f.shape == (FP_DIM,)
        assert np.allclose(f[:K], (m - m.mean()) / m.std()) and f[K] == pytest.approx(m.mean()) and f[K + 1] == pytest.approx(m.std())
        flat = m33.fingerprint(np.full(K, 0.7))
        assert np.all(flat[:K] == 0.0) and flat[K + 1] == pytest.approx(0.0, abs=1e-9)   # 六个 0.7 的 std 是 1e-16 级

    def test_context_for_fp_budget_vector(self):
        keys = [f"k{j}" for j in range(12)]
        slices = ["Cricket_00", "Cricket_01", "Handwriting_00"]
        cache = _cache(slices, keys)
        probes = [f"k{j}" for j in range(K)]
        raw = m33.fp_raw(cache, slices, probes, budget=60)
        mu, sd = m33.fp_standardizer(raw, ["Cricket_00", "Cricket_01"])
        tid = pq.engine_task_id("Handwriting_00", 290)
        ctx = m33.context_for(m33.FP_ARMS["fp_diff_hard"], [tid], pq.BUDGETS, pq.PQ_DATASETS,
                              fp=raw, fp_mu=mu, fp_sd=sd)[tid]
        assert ctx.shape == (CTX_DIM,) and ctx.dtype == np.float32
        assert np.allclose(ctx[:FP_DIM], (raw["Handwriting_00"] - mu) / sd)
        assert np.array_equal(ctx[FP_DIM:], pq.ctx_vector(290, pq.BUDGETS))
        with pytest.raises(ValueError):
            m33.context_for(m33.FP_ARMS["fp_diff_hard"], [tid], pq.BUDGETS, pq.PQ_DATASETS)

    def test_train_fp_diff_hard_on_smoke_dir(self, tmp_path):
        # 冒烟合成把 n_cls // 2 标成 shared：要凑齐 K=6 条探针需 n_cls ≥ 12
        m33.synth_smoke_inputs(tmp_path, datasets=("Handwriting", "Cricket"), n_slices=3, n_cls=16,
                               budgets=pq.BUDGETS)
        m33.cmd_train(tmp_path, arms=["fp_diff_hard"], folds=["Handwriting"], seeds=[0], resume=False,
                      device="cpu", epochs=2, settf_windows=2, smoke=True, encoder_mode="binned")
        meta = torch.load(tmp_path / "models" / "Handwriting" / "fp_diff_hard_s0.pt", map_location="cpu",
                          weights_only=True)["meta"]
        assert meta["ctx_kind"] == "fp_budget" and meta["ctx_dim"] == CTX_DIM and meta["head"] == "diff"
        assert len(meta["fp_probes"]) == K and meta["fp_budget"] == 60
        assert len(meta["fp_mu"]) == FP_DIM and len(meta["fp_sd"]) == FP_DIM
        run = json.loads((tmp_path / "runs" / "Handwriting" / "fp_diff_hard_s0.json").read_text(encoding="utf-8"))
        assert run["settings"]["ctx_kind"] == "fp_budget" and run["cross"] is not None
        assert run["settings"]["fp_probes"] == meta["fp_probes"]
        model, _ = g34.load_pq_model(tmp_path / "models" / "Handwriting" / "fp_diff_hard_s0.pt",
                                     expect_fold="Handwriting", expect_budgets=pq.BUDGETS)
        assert model.ctx_dim == CTX_DIM


# ================================================================ B. 59 号
def _setup():
    keys = [f"k{j}" for j in range(12)]
    slices = ["Handwriting_00", "Handwriting_01", "Cricket_00", "Cricket_01", "UWaveGestureLibrary_00"]
    cache = _cache(slices, keys)
    probes = [f"k{j}" for j in range(K)]
    raw = m33.fp_raw(cache, slices, probes, budget=60)
    mu, sd = m33.fp_standardizer(raw, ["Cricket_00", "Cricket_01", "UWaveGestureLibrary_00"])
    return keys, slices, cache, probes, raw, mu, sd


class TestFpCtxRows:
    def test_normal_rows_match_context_for(self):
        g = _g59()
        keys, slices, cache, probes, raw, mu, sd = _setup()
        rows = g.fp_ctx_rows(raw, ["Handwriting_00"], mu, sd)
        tid = pq.engine_task_id("Handwriting_00", 120)
        ref = m33.context_for(m33.FP_ARMS["fp_diff_hard"], [tid], pq.BUDGETS, pq.PQ_DATASETS, fp=raw, fp_mu=mu, fp_sd=sd)[tid]
        assert np.array_equal(rows[("Handwriting_00", 120)], ref)

    def test_cross_and_zero_rows(self):
        g = _g59()
        keys, slices, cache, probes, raw, mu, sd = _setup()
        cross = g.fp_ctx_rows(raw, ["Handwriting_00"], mu, sd, cross_map={"Handwriting_00": "Cricket_01"})
        donor = g.fp_ctx_rows(raw, ["Cricket_01"], mu, sd)
        assert np.array_equal(cross[("Handwriting_00", 60)][:FP_DIM], donor[("Cricket_01", 60)][:FP_DIM])
        zero = g.fp_ctx_rows(raw, ["Handwriting_00"], mu, sd, zero=True)
        assert not zero[("Handwriting_00", 60)][:FP_DIM].any() and zero[("Handwriting_00", 60)][FP_DIM:].sum() == 1.0


class TestNonprobeView:
    def test_filter_removes_probes_from_rankings_and_rows(self):
        g = _g59()
        rk = {"60": {"Handwriting_00": {"fp_diff_hard": [f"k{j}" for j in range(12)]}}}
        rows = [{"dataset": "Handwriting", "task_id": "Handwriting_00", "iters": 60, "cls_key": f"k{j}", "r_prime": 0.9 - 0.03 * j,
                 "z": 0.0, "origin": "shared"} for j in range(12)]
        probes = [f"k{j}" for j in range(K)]
        rk2, rows2 = g.nonprobe_view(rk, rows, probes)
        assert rk2["60"]["Handwriting_00"]["fp_diff_hard"] == [f"k{j}" for j in range(K, 12)]
        assert {r["cls_key"] for r in rows2} == {f"k{j}" for j in range(K, 12)}
        assert all(r["cls_key"] not in probes for r in rows2)


class TestProbeTop3Reference:
    def test_probe6_top3_regret(self):
        g = _g59()
        keys, slices, cache, probes, raw, mu, sd = _setup()
        # 探针按 @60 实测取前 3，regret@3 = 该预算下目录 oracle − 前 3 里的最好真值
        r = g.probe_top3_regret(cache, "Handwriting_00", probes, keys, budget=290)
        m60 = {p: cache[("Handwriting_00", p, 60)]["r_prime"] for p in probes}
        top3 = sorted(probes, key=lambda p: (-m60[p], p))[:3]
        oracle = max(cache[("Handwriting_00", k, 290)]["r_prime"] for k in keys)
        assert r == pytest.approx(oracle - max(cache[("Handwriting_00", p, 290)]["r_prime"] for p in top3))


class TestGatesS1Fp:
    def _g(self, d1=0.05, p1=0.01, reg_fp=0.020, reg_blind=0.030, p2=0.01, fp_all=0.015, probe=0.020, p3=0.01):
        deltas = {"fp_diff_hard": {"acc_top10": {"mean_delta": d1, "p": p1}},
                  "fp_concat_hard": {"acc_top10": {"mean_delta": 0.0, "p": 0.8}}}
        v2 = {"mean_a": reg_blind, "mean_b": reg_fp, "p": p2}
        v3 = {"mean_a": probe, "mean_b": fp_all, "p": p3}
        return deltas, v2, v3

    def test_codes(self):
        g = _g59()
        assert g.verdict_s1fp(g.gates_s1fp(*self._g())) == "FP-ALIVE"
        assert g.verdict_s1fp(g.gates_s1fp(*self._g(fp_all=0.025, p3=0.5))) == "FP-EXTRAPOLATES"
        assert g.verdict_s1fp(g.gates_s1fp(*self._g(reg_fp=0.031, p2=0.7))) == "FP-USED-NO-GAIN"
        assert g.verdict_s1fp(g.gates_s1fp(*self._g(d1=0.02))) == "FP-DEAD"
        assert g.verdict_s1fp(g.gates_s1fp(*self._g(p1=0.3))) == "FP-DEAD"

    def test_value_gates_need_direction_and_p(self):
        g = _g59()
        gt = g.gates_s1fp(*self._g(reg_fp=0.035, p2=0.01))
        assert not gt["GF2"]["pass"]
        gt = g.gates_s1fp(*self._g(fp_all=0.030, probe=0.020, p3=0.001))
        assert not gt["GF3"]["pass"]
