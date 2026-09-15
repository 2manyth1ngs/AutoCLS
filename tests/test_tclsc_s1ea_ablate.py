"""R 检验 §9 消融补充——先于实现。fwb 臂（field 头 + 只有预算 one-hot）在 33 / 60 两条训练器上可训可排；推理侧 zero_all / noise 行；67 号读数结构。"""
from __future__ import annotations

import importlib
import json

import numpy as np
import pytest
import torch

from autocls.data.mech_features import RESPONSE_PROFILE_COLS
from autocls.space.cls_space import CLSSpace
from scripts import pq_common as pq

m33 = importlib.import_module("scripts.33_pq_comparator_loo")
SPACE = CLSSpace.from_yaml()
EA_COLS = tuple(RESPONSE_PROFILE_COLS)
PHI_COLS = tuple(m33.PHI_COLS)
ALL_COLS = EA_COLS + PHI_COLS


def _g67():
    return importlib.import_module("scripts.67_tclsc_s1ea_ablate")


def _ucr_rows(n_datasets=4, n_cls=20, seed=0):
    rng = np.random.default_rng(seed)
    cands = [SPACE.sample_random(rng) for _ in range(n_cls)]
    from autocls.labeling import cls_key
    rows = []
    for i in range(n_datasets):
        name = f"DS{i:02d}"
        for j, c in enumerate(cands):
            for b in pq.BUDGETS:
                rows.append({"dataset": name, "task_id": f"{name}_00", "iters": int(b), "cls_key": cls_key(c),
                             "origin": "shared" if j < 10 else "random", "cls": c.to_dict(), "error": None,
                             "r_prime": float(0.9 - 0.02 * j + 0.01 * rng.standard_normal())})
    return rows


class TestFwbArms:
    def test_arm_specs(self):
        a = m33.EA_ARMS["fwb_hard"]
        assert a.head == "field" and a.ctx_kind == "budget" and a.task_branch == "ctx" and a.ctx_cols is None and a.head_rank is None
        assert m33.ctx_dim_for(a, pq.BUDGETS, pq.PQ_DATASETS) == 3
        g60 = importlib.import_module("scripts.60_tclsc_s1ucr")
        u = g60.ARMS["u_fwb_hard"]
        assert u.head == "field" and u.ctx_kind == "budget" and u.seeds == (0, 1, 2)

    def test_smoke_train_fwb_33(self, tmp_path):
        m33.synth_smoke_inputs(tmp_path, datasets=("Handwriting", "Cricket"), n_slices=3, n_cls=4, budgets=pq.BUDGETS)
        m33.cmd_train(tmp_path, arms=["fwb_hard"], folds=["Handwriting"], seeds=[0], resume=False, device="cpu",
                      epochs=2, settf_windows=2, smoke=True, encoder_mode="binned")
        meta = torch.load(tmp_path / "models" / "Handwriting" / "fwb_hard_s0.pt", map_location="cpu", weights_only=True)["meta"]
        assert meta["head"] == "field" and meta["ctx_dim"] == 3 and meta["ctx_kind"] == "budget"

    def test_train_and_rank_fwb_60(self, tmp_path):
        g60 = importlib.import_module("scripts.60_tclsc_s1ucr")
        rows = _ucr_rows()
        rng = np.random.default_rng(1)
        phi = {d: {"task_id": f"{d}_00", "dataset": d, **{c: float(rng.standard_normal()) for c in PHI_COLS}} for d in sorted({r["dataset"] for r in rows})}
        folds = [["DS02", "DS03"], ["DS00", "DS01"]]
        g60.train_fold(rows, phi, folds, 0, arm="u_fwb_hard", seed=0, outdir=tmp_path, device="cpu", epochs=2)
        meta = torch.load(tmp_path / "models" / "fold0" / "u_fwb_hard_s0.pt", map_location="cpu", weights_only=True)["meta"]
        assert meta["ctx_dim"] == 3 and meta["head"] == "field"
        rk = g60.rank_fold(rows, phi, folds, 0, arms=["u_fwb_hard"], seeds=[0], outdir=tmp_path)
        keys = sorted({r["cls_key"] for r in rows})
        assert sorted(rk["normal"]["60"]["DS02_00"]["u_fwb_hard"]) == keys


class TestInferenceRows:
    def test_zero_all_and_noise(self):
        g = _g67()
        slices = ["Cricket_00", "Cricket_01"]
        rng = np.random.default_rng(0)
        table = {s: {c: float(rng.standard_normal()) for c in ALL_COLS} for s in slices}
        mu, sd = np.zeros(37), np.ones(37)
        z = g.ablate_rows(table, ALL_COLS, slices, mu, sd, mode="zero_all")
        assert np.all(z[("Cricket_00", 60)][:37] == 0) and z[("Cricket_00", 60)][37:].tolist() == pq.ctx_vector(60, pq.BUDGETS).tolist()
        n1 = g.ablate_rows(table, ALL_COLS, slices, mu, sd, mode="noise", seed=0)
        n2 = g.ablate_rows(table, ALL_COLS, slices, mu, sd, mode="noise", seed=0)
        n3 = g.ablate_rows(table, ALL_COLS, slices, mu, sd, mode="noise", seed=1)
        assert np.allclose(n1[("Cricket_00", 60)], n2[("Cricket_00", 60)]) and not np.allclose(n1[("Cricket_00", 60)][:37], n3[("Cricket_00", 60)][:37])
        assert np.allclose(n1[("Cricket_00", 60)][:37], n1[("Cricket_00", 290)][:37])     # 同切片各预算同一噪声向量
        assert abs(float(np.std(np.concatenate([n1[k][:37] for k in n1]))) - 1.0) < 0.5
        with pytest.raises(ValueError):
            g.ablate_rows(table, ALL_COLS, slices, mu, sd, mode="shuffle")
        assert g.MODES == ("zero_all", "noise")
