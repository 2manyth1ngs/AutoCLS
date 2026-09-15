"""U 检验（UEA30 靶外 23 靶官方口径名单）——先于实现。覆盖：all8 训练（源池全部切片、无折）、官方 φ 部分列、目录轨迹 + 限域三名单 SH、
picks 生成规则、门与读数码。零 GPU、不读真实缓存。"""
from __future__ import annotations

import importlib
import json

import numpy as np
import pytest
import torch

from autocls.data.mech_features import REP_PROFILE_COLS, rep_cols_at
from scripts import pq_common as pq

m33 = importlib.import_module("scripts.33_pq_comparator_loo")
PHI_COLS = tuple(m33.PHI_COLS)


def _g72():
    return importlib.import_module("scripts.72_tclsc_s1et_uea30")


class TestPhiPartial:
    def test_phi_partial_cols(self):
        g = _g72()
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 32, 3)).astype(np.float32); y = np.repeat(np.arange(4), 10)
        row = g.phi_partial(X, y)
        assert set(PHI_COLS) <= set(row)
        assert row["base_mean"] is None and row["base_band"] is None                 # 无 iters0 基线 → None → 训练均值填补
        assert row["log10_C"] == pytest.approx(np.log10(3)) and row["k"] == 4


class TestAll8Train:
    def test_train_all8_smoke(self, tmp_path):
        g = _g72()
        m33.synth_smoke_inputs(tmp_path, datasets=("Handwriting", "Cricket"), n_slices=3, n_cls=6, budgets=pq.BUDGETS)
        slices = [f"{d}_{i:02d}" for d in ("Handwriting", "Cricket") for i in range(3)]
        rng = np.random.default_rng(1)
        table = {s: {"task_id": s, "dataset": pq.dataset_of(s), **{c: float(rng.standard_normal()) for c in rep_cols_at(60) + PHI_COLS}} for s in slices}
        g.train_all8(tmp_path, table, seeds=[0], device="cpu", epochs=2, arms=("blind_diff_hard", "ft60_hard"))
        for a in ("blind_diff_hard", "ft60_hard"):
            assert (tmp_path / "models" / "all8" / f"{a}_s0.pt").exists()
        meta = torch.load(tmp_path / "models" / "all8" / "ft60_hard_s0.pt", map_location="cpu", weights_only=True)["meta"]
        assert meta["fold"] == "all8" and meta["head"] == "field" and len(meta["ctx_mu"]) == 34 and list(meta["ctx_cols"]) == list(rep_cols_at(60) + PHI_COLS)
        assert g.ALL8_FOLD == "all8"


def _runs(target="T"):
    rows = []
    keys = [f"k{i}" for i in range(11)] + [f"n{i}" for i in range(3)]
    for i, k in enumerate(keys):
        accs = [0.3 + 0.02 * i, 0.35 + 0.02 * i, 0.4 + 0.02 * i]
        rows.append({"dataset": target, "stage": "s1_catalog", "cls_key": k, "seed": 0, "iters": 200, "checkpoints": [50, 100, 200], "acc": accs[-1],
                     "arm": ("ggs" if k == "k10" else "shared"), "cls": {}, "error": None,
                     "checkpoint_results": [{"iters": b, "acc": a} for b, a in zip((50, 100, 200), accs)]})
    rows.append({"dataset": target, "stage": "s1_catalog", "cls_key": "t0", "twin_of": "k0", "seed": 0, "iters": 200, "checkpoints": [50, 100, 200], "acc": 0.9,
                 "arm": "twin", "cls": {}, "error": None, "checkpoint_results": [{"iters": b, "acc": 0.9} for b in (50, 100, 200)]})
    rows.append({"dataset": target, "stage": "s2_finals", "cls_key": "k10", "seed": 1, "acc": 0.6, "error": None})
    rows.append({"dataset": target, "stage": "s2_finals", "cls_key": "k10", "seed": 2, "acc": 0.62, "error": None})
    return rows


class TestSimulate:
    def test_trust_keys_and_sh(self):
        g = _g72()
        rows = _runs()
        traj, finals, budgets, keys_off = g.target_catalog(rows, "T")
        assert set(keys_off) == {f"k{i}" for i in range(11)} and "t0" not in keys_off and "n0" not in traj or "n0" in traj
        assert "t0" not in keys_off and budgets == (50, 100, 200) and finals["k10"] == [0.6, 0.6, 0.62]
        order = ["n2", "k10", "k9", "k1", "k2", "k3", "k4", "k5"]
        r = g.sh_trust(order, traj, keys_off, budgets)
        assert r["champion"] == "k10" and r["list"][0] == "k10" and "n2" not in r["list"]
        picks = g.picks_for({"T": {"prior": "k10", "blind_tclsc": "k9", "et_tclsc": "k9"}}, {("T", "k10"): {1, 2}})
        assert picks == {"T": {"SH6-3-3_blind_tclsc": "k9", "SH6-3-3_et_tclsc": "k9"}}

    def test_gates_codes(self):
        g = _g72()
        v = lambda a, b, p: {"mean_a": a, "mean_b": b, "p": p}
        assert g.verdict_uea30(g.gates_uea30(0.740, 0.730, 17, 0.735, 12, 12, v(0.73, 0.74, 0.01))) == "UEA30-ET-WIN+PRIOR"
        assert g.verdict_uea30(g.gates_uea30(0.740, 0.730, 17, 0.735, 12, 12, v(0.73, 0.74, 0.3))) == "UEA30-ET-BETTER+PRIOR"
        assert g.verdict_uea30(g.gates_uea30(0.740, 0.730, 17, 0.745, 12, 12, v(0.73, 0.74, 0.01))) == "UEA30-ET-WIN-PRIOR"
        assert g.verdict_uea30(g.gates_uea30(0.732, 0.730, 17, 0.735, 12, 12, v(0.73, 0.74, 0.01))) == "UEA30-ET-FLAT+PRIOR"
        assert g.verdict_uea30(g.gates_uea30(0.740, 0.730, 10, 0.735, 12, 12, v(0.73, 0.74, 0.01))) == "UEA30-ET-FLAT+PRIOR"
