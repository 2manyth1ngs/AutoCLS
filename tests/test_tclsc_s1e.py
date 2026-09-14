"""WP16-TCLSC 阶段 1 真 E 臂——先于实现（铁律 5）。协议 results/tclsc_s1e/PROTOCOL.md。

A. `scripts/33` `E_ARMS`：e_concat_hard / e_diff_hard（E+ctx, settf, fold, budget, S1, 3 种子, 铁律 6 合规）；arm_spec 解析。
B. `scripts/62_tclsc_s1e.py`：E 行（normal / cross / zero；3 种子各自的 SetTF）、E 诊断（跨集/集内余弦、范数）、
   单折排名（toy SetTF + toy 窗口）、门与读数码。
零 GPU、不读真实缓存。
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from autocls.comparator.tclsc_model import TCLSC
from autocls.labeling import cls_key
from autocls.space.cls_space import CLSSpace
from autocls.task_embed.set_transformer import TaskSetEncoder
from scripts import pq_common as pq

m33 = importlib.import_module("scripts.33_pq_comparator_loo")
SPACE = CLSSpace.from_yaml()
E_NAMES = ("e_concat_hard", "e_diff_hard")


def _g62():
    return importlib.import_module("scripts.62_tclsc_s1e")


def _toy_model(head, seed=0, nz=8):
    torch.manual_seed(seed)
    enc = TaskSetEncoder(dim_input=8, nz=nz)
    return TCLSC(cls_dim=116, set_encoder=enc, embedding_dim=8, fz=8, ctx_dim=3, task_branch="E+ctx", head=head).eval()


def _windows(slices, seed=0):
    rng = np.random.default_rng(seed)
    return {s: torch.as_tensor(rng.standard_normal((4, 6, 8)).astype(np.float32)) for s in slices}


class TestEArms:
    def test_table(self):
        assert set(m33.E_ARMS) == set(E_NAMES)
        assert not (set(m33.E_ARMS) & (set(m33.ARMS) | set(m33.L3_ARMS) | set(m33.PHI_ARMS) | set(m33.FP_ARMS)))
        for name, s in m33.E_ARMS.items():
            assert s.split == "S1" and s.task_branch == "E+ctx" and s.encoder == "settf" and s.e_source == "fold"
            assert s.ctx_kind == "budget" and s.label_mode == "hard" and s.seeds == (0, 1, 2)
            assert s.iron_law_6_compliant is True and s.head == name.split("_")[1]
        assert m33.arm_spec("e_diff_hard") is m33.E_ARMS["e_diff_hard"]
        assert "e_diff_hard" in m33.CLI_ARMS


class TestERows:
    def test_e_rows_normal_cross_zero(self):
        g = _g62()
        m = _toy_model("diff")
        test = ["Handwriting_04", "Handwriting_05"]
        train = ["Cricket_00", "UWaveGestureLibrary_00"]
        w = _windows(test + train)
        normal = g.e_rows(m, w, test)
        assert set(normal) == set(test) and all(v.shape == (8,) for v in normal.values())
        with torch.no_grad():
            ref = m.encode_tasks([w["Handwriting_04"]])[0]
        assert torch.allclose(normal["Handwriting_04"], ref)
        cmap = {"Handwriting_04": "Cricket_00", "Handwriting_05": "UWaveGestureLibrary_00"}
        cross = g.e_rows(m, w, test, cross_map=cmap)
        assert torch.allclose(cross["Handwriting_04"], g.e_rows(m, w, ["Cricket_00"])["Cricket_00"])
        zero = g.e_rows(m, w, test, zero=True)
        assert not zero["Handwriting_04"].any()

    def test_e_diagnostics(self):
        g = _g62()
        E = {"A_00": np.array([1.0, 0.0]), "A_01": np.array([0.9, 0.1]), "B_00": np.array([0.0, 1.0]), "B_01": np.array([0.1, 0.9])}
        d = g.e_diagnostics(E)
        assert d["cos_within"] > 0.9 and d["cos_cross"] < 0.3 and d["n_slices"] == 4
        assert d["norm_median"] == pytest.approx(np.median([np.linalg.norm(v) for v in E.values()]))
        # 全同向量 → 塌缩形态
        flat = g.e_diagnostics({k: np.array([1.0, 1.0]) for k in E})
        assert flat["cos_cross"] == pytest.approx(1.0)


class TestRankOneFold:
    def test_rank_target_with_toy_settf(self):
        g = _g62()
        rng = np.random.default_rng(3)
        cs = [SPACE.sample_random(rng) for _ in range(5)]
        cat = {"Handwriting_04": [("shared", c, cls_key(c)) for c in cs], "Handwriting_05": [("shared", c, cls_key(c)) for c in cs]}
        test = list(cat)
        train = ["Cricket_00", "Cricket_01", "UWaveGestureLibrary_00"]
        w = _windows(test + train)
        models = [(_toy_model("diff", s), {"encoder_mode": "binned", "cls_dim": 116, "head": "diff"}) for s in range(2)]
        out = g.rank_target(models, cat, w, train_slices=train, budgets=pq.BUDGETS, cross_seeds=(0, 1))
        keys = sorted(cls_key(c) for c in cs)
        for b in pq.BUDGETS:
            for s in test:
                assert sorted(out["normal"][str(b)][s]["arm"]) == keys
                assert sorted(out["zero"][str(b)][s]["arm"]) == keys
        assert set(out["cross"]) == {"0", "1"}
        assert all(pq.dataset_of(v) != "Handwriting" for v in out["cross"]["0"]["map"].values())
        assert set(out["diag"]) >= {"cos_within", "cos_cross", "norm_median"}


class TestGatesE:
    def _g(self, diag_cross=0.5, diag_within=0.8, d1=0.05, p1=0.01, reg_e=0.020, reg_blind=0.029, p2=0.01, d_cc=0.0, p_cc=0.9):
        diag = {"cos_cross": diag_cross, "cos_within": diag_within}
        deltas = {"e_diff_hard": {"acc_top10": {"mean_delta": d1, "p": p1}},
                  "e_concat_hard": {"acc_top10": {"mean_delta": d_cc, "p": p_cc}}}
        value = {"mean_a": reg_blind, "mean_b": reg_e, "p": p2}
        return diag, deltas, value

    def test_codes(self):
        g = _g62()
        assert g.verdict_e(g.gates_e(*self._g())) == "E-ALIVE"
        assert g.verdict_e(g.gates_e(*self._g(reg_e=0.031, p2=0.6))) == "E-USED-NO-GAIN"
        assert g.verdict_e(g.gates_e(*self._g(d1=0.01))) == "E-INERT"
        assert g.verdict_e(g.gates_e(*self._g(diag_cross=0.995, diag_within=0.999))) == "E-COLLAPSED"
        assert g.verdict_e(g.gates_e(*self._g(diag_cross=0.98, diag_within=0.985))) == "E-COLLAPSED"   # 集内−跨集 < 0.01

    def test_control_flag(self):
        g = _g62()
        gt = g.gates_e(*self._g(d_cc=0.06, p_cc=0.01))
        assert gt["GE3"]["pass"] is False and gt["GE1"]["pass"] is True
