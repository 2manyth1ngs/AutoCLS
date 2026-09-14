"""WP16-TCLSC 阶段 1 φ 臂——先于实现（铁律 5）。协议 results/tclsc_s1phi/PROTOCOL.md。

A. `scripts/33`：`PHI_COLS` 与 40 号一致；`PHI_ARMS` 表（S1、3 种子、铁律 6 合规）；`ctx_kind="phi_budget"` 的维度与向量
   （折内标准化 φ ⊕ 预算 one-hot）；`phi_standardizer` 只用给定切片；`PQContext(phi_file=)`；冒烟目录训 `phi_diff_hard` →
   meta 记 ctx_kind / phi_mu / phi_sd / phi_cols，ctx_dim 15，cross 块出块。
B. `scripts/58_tclsc_s1phi.py`：φ ctx 行（normal / cross / zero）、跨数据集供体映射、门与读数码。
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

PHI_COLS = ("log10_T", "log10_C", "k", "log10_N", "base_mean", "base_band",
            "spec_conc", "hf_ratio", "acf_tau", "ex_kurt", "chan_cv", "samp_cv")
CTX_DIM = len(PHI_COLS) + len(pq.BUDGETS)      # 12 + 3
PHI_NAMES = ("blind_diff_hard", "phi_concat_hard", "phi_diff_hard", "phi_diff_zdiff")


def _g58():
    return importlib.import_module("scripts.58_tclsc_s1phi")


def _phi_table(slices, seed=0):
    rng = np.random.default_rng(seed)
    return {s: {"task_id": s, "dataset": pq.dataset_of(s), **{c: float(rng.standard_normal()) for c in PHI_COLS}}
            for s in slices}


def _write_phi(path, table):
    with open(path, "w", encoding="utf-8") as fh:
        for r in table.values():
            fh.write(json.dumps(r) + "\n")
    return path


# ================================================================ A. 33 号
class TestPhiArms:
    def test_phi_cols_match_script40(self):
        g40 = importlib.import_module("scripts.40_task_feature_scorer")
        assert tuple(m33.PHI_COLS) == tuple(g40.PHI_COLS) == PHI_COLS

    def test_table_shape(self):
        assert set(m33.PHI_ARMS) == set(PHI_NAMES)
        assert not (set(m33.PHI_ARMS) & (set(m33.ARMS) | set(m33.L3_ARMS)))
        for name, spec in m33.PHI_ARMS.items():
            assert spec.split == "S1" and spec.seeds == (0, 1, 2) and spec.iron_law_6_compliant is True
            assert spec.encoder is None and spec.e_source is None
        b = m33.PHI_ARMS["blind_diff_hard"]
        assert b.task_branch == "none" and b.ctx_kind == "none" and b.head == "diff" and b.label_mode == "hard"
        for name in ("phi_concat_hard", "phi_diff_hard", "phi_diff_zdiff"):
            s = m33.PHI_ARMS[name]
            assert s.task_branch == "ctx" and s.ctx_kind == "phi_budget"
            assert s.head == name.split("_")[1] and s.label_mode == name.split("_")[2]
        assert m33.arm_spec("phi_diff_hard") is m33.PHI_ARMS["phi_diff_hard"]

    def test_ctx_dim_phi_budget(self):
        assert m33.ctx_dim_for(m33.PHI_ARMS["phi_diff_hard"], pq.BUDGETS, pq.PQ_DATASETS) == CTX_DIM
        assert m33.ctx_dim_for(m33.PHI_ARMS["blind_diff_hard"], pq.BUDGETS, pq.PQ_DATASETS) == 0

    def test_standardizer_uses_only_given_slices(self):
        table = _phi_table(["Handwriting_00", "Handwriting_01", "Cricket_00", "Cricket_01"])
        mu, sd = m33.phi_standardizer(table, ["Handwriting_00", "Handwriting_01"], PHI_COLS)
        ref = np.stack([[table[s][c] for c in PHI_COLS] for s in ("Handwriting_00", "Handwriting_01")])
        assert np.allclose(mu, ref.mean(0)) and np.allclose(sd, np.maximum(ref.std(0), 1e-6))
        assert mu.shape == (12,) and sd.shape == (12,)

    def test_context_for_phi_budget_vector(self):
        table = _phi_table(["Handwriting_00", "Cricket_00"])
        mu, sd = m33.phi_standardizer(table, ["Cricket_00"], PHI_COLS)
        tid = pq.engine_task_id("Handwriting_00", 120)
        ctx = m33.context_for(m33.PHI_ARMS["phi_diff_hard"], [tid], pq.BUDGETS, pq.PQ_DATASETS,
                              phi=table, phi_mu=mu, phi_sd=sd)[tid]
        assert ctx.shape == (CTX_DIM,) and ctx.dtype == np.float32
        expect = (np.array([table["Handwriting_00"][c] for c in PHI_COLS]) - mu) / sd
        assert np.allclose(ctx[:12], expect)
        assert np.array_equal(ctx[12:], pq.ctx_vector(120, pq.BUDGETS))

    def test_context_for_phi_requires_table(self):
        tid = pq.engine_task_id("Handwriting_00", 60)
        with pytest.raises(ValueError):
            m33.context_for(m33.PHI_ARMS["phi_diff_hard"], [tid], pq.BUDGETS, pq.PQ_DATASETS)

    def test_train_phi_diff_hard_on_smoke_dir(self, tmp_path):
        m33.synth_smoke_inputs(tmp_path, datasets=("Handwriting", "Cricket"), n_slices=3, n_cls=4,
                               budgets=pq.BUDGETS)
        slices = [f"{d}_{i:02d}" for d in ("Handwriting", "Cricket") for i in range(3)]
        phi_file = _write_phi(tmp_path / "phi.jsonl", _phi_table(slices))
        m33.cmd_train(tmp_path, arms=["phi_diff_hard"], folds=["Handwriting"], seeds=[0], resume=False,
                      device="cpu", epochs=2, settf_windows=2, smoke=True, encoder_mode="binned",
                      phi_file=phi_file)
        blob = torch.load(tmp_path / "models" / "Handwriting" / "phi_diff_hard_s0.pt", map_location="cpu",
                          weights_only=True)
        meta = blob["meta"]
        assert meta["head"] == "diff" and meta["ctx_dim"] == CTX_DIM and meta["ctx_kind"] == "phi_budget"
        assert list(meta["phi_cols"]) == list(PHI_COLS) and len(meta["phi_mu"]) == 12 and len(meta["phi_sd"]) == 12
        # 标准化只用训练切片（Cricket_00..02）：与手算一致（铁律 6）
        mu, sd = m33.phi_standardizer(_phi_table(slices), [s for s in slices if s.startswith("Cricket")], PHI_COLS)
        assert np.allclose(meta["phi_mu"], mu) and np.allclose(meta["phi_sd"], sd)
        run = json.loads((tmp_path / "runs" / "Handwriting" / "phi_diff_hard_s0.json").read_text(encoding="utf-8"))
        assert run["settings"]["ctx_kind"] == "phi_budget" and run["cross"] is not None
        model, meta2 = g34.load_pq_model(tmp_path / "models" / "Handwriting" / "phi_diff_hard_s0.pt",
                                         expect_fold="Handwriting", expect_budgets=pq.BUDGETS)
        assert model.head == "diff" and model.ctx_dim == CTX_DIM

    def test_train_phi_arm_without_phi_file_fails_fast(self, tmp_path):
        m33.synth_smoke_inputs(tmp_path, datasets=("Handwriting", "Cricket"), n_slices=3, n_cls=4,
                               budgets=pq.BUDGETS)
        with pytest.raises((FileNotFoundError, ValueError, SystemExit)):
            m33.cmd_train(tmp_path, arms=["phi_diff_hard"], folds=["Handwriting"], seeds=[0], resume=False,
                          device="cpu", epochs=2, settf_windows=2, smoke=True, encoder_mode="binned",
                          phi_file=tmp_path / "missing.jsonl")


# ================================================================ B. 58 号
class TestPhiCtxRows:
    def _setup(self):
        slices = ["Handwriting_04", "Handwriting_05", "Cricket_00", "Cricket_01", "UWaveGestureLibrary_00"]
        table = _phi_table(slices)
        mu, sd = m33.phi_standardizer(table, ["Cricket_00", "Cricket_01", "UWaveGestureLibrary_00"], PHI_COLS)
        return slices, table, mu, sd

    def test_normal_rows_match_context_for(self):
        g = _g58()
        slices, table, mu, sd = self._setup()
        rows = g.phi_ctx_rows(table, ["Handwriting_04"], mu, sd)
        tid = pq.engine_task_id("Handwriting_04", 290)
        ref = m33.context_for(m33.PHI_ARMS["phi_diff_hard"], [tid], pq.BUDGETS, pq.PQ_DATASETS,
                              phi=table, phi_mu=mu, phi_sd=sd)[tid]
        assert np.array_equal(rows[("Handwriting_04", 290)], ref)
        assert set(rows) == {("Handwriting_04", b) for b in pq.BUDGETS}

    def test_cross_rows_use_donor_phi_and_keep_budget(self):
        g = _g58()
        slices, table, mu, sd = self._setup()
        cmap = {"Handwriting_04": "Cricket_01"}
        rows = g.phi_ctx_rows(table, ["Handwriting_04"], mu, sd, cross_map=cmap)
        donor = g.phi_ctx_rows(table, ["Cricket_01"], mu, sd)
        assert np.array_equal(rows[("Handwriting_04", 60)][:12], donor[("Cricket_01", 60)][:12])
        assert np.array_equal(rows[("Handwriting_04", 60)][12:], pq.ctx_vector(60, pq.BUDGETS))

    def test_zero_rows(self):
        g = _g58()
        slices, table, mu, sd = self._setup()
        rows = g.phi_ctx_rows(table, ["Handwriting_04"], mu, sd, zero=True)
        assert not rows[("Handwriting_04", 120)][:12].any() and rows[("Handwriting_04", 120)][12:].sum() == 1.0

    def test_cross_slice_map_donors_from_other_datasets(self):
        g = _g58()
        test = ["Handwriting_04", "Handwriting_05"]
        train = ["Cricket_00", "Cricket_01", "UWaveGestureLibrary_00", "Handwriting_00"]
        m1 = g.cross_slice_map(test, train, seed=0)
        assert set(m1) == set(test)
        assert all(pq.dataset_of(v) != "Handwriting" and v in train for v in m1.values())
        assert m1 == g.cross_slice_map(test, train, seed=0)


class TestGatesS1Phi:
    FOLDS = ("HW", "UM", "PS", "UW", "ET", "CR", "HM")

    def _inputs(self, d_phi, p_phi, d_cc=0.0, p_cc=0.9, reg_phi=0.030, reg_blind=0.033, p_reg=0.9):
        deltas = {"phi_diff_hard": {"acc_top10": {"mean_delta": d_phi, "p": p_phi}},
                  "phi_concat_hard": {"acc_top10": {"mean_delta": d_cc, "p": p_cc}}}
        value = {"a": "blind_diff_hard", "b": "phi_diff_hard", "metric": "regret3", "mean_a": reg_blind,
                 "mean_b": reg_phi, "p": p_reg}
        return deltas, value

    def test_gp1_gp2_thresholds_and_codes(self):
        g = _g58()
        gt = g.gates_s1phi(*self._inputs(0.05, 0.01, reg_phi=0.025, p_reg=0.01))
        assert gt["GP1"]["pass"] and gt["GP2"]["pass"] and gt["GP3"]["pass"]
        assert g.verdict_s1phi(gt) == "PHI-ALIVE"
        gt = g.gates_s1phi(*self._inputs(0.05, 0.01, reg_phi=0.035, p_reg=0.6))
        assert gt["GP1"]["pass"] and not gt["GP2"]["pass"] and g.verdict_s1phi(gt) == "PHI-USED-NO-GAIN"
        gt = g.gates_s1phi(*self._inputs(0.02, 0.01, reg_phi=0.020, p_reg=0.01))
        assert not gt["GP1"]["pass"] and g.verdict_s1phi(gt) == "PHI-DEAD"
        gt = g.gates_s1phi(*self._inputs(0.05, 0.2))
        assert not gt["GP1"]["pass"] and g.verdict_s1phi(gt) == "PHI-DEAD"

    def test_gp3_control_flags_concat_positive(self):
        g = _g58()
        gt = g.gates_s1phi(*self._inputs(0.05, 0.01, d_cc=0.06, p_cc=0.01))
        assert gt["GP3"]["pass"] is False           # concat 也正：只报不进读数码
        assert g.verdict_s1phi(gt) in ("PHI-ALIVE", "PHI-USED-NO-GAIN")

    def test_value_gate_requires_lower_regret_not_just_p(self):
        g = _g58()
        gt = g.gates_s1phi(*self._inputs(0.05, 0.01, reg_phi=0.040, reg_blind=0.033, p_reg=0.01))
        assert not gt["GP2"]["pass"]                 # 显著但方向反（regret 更高）
