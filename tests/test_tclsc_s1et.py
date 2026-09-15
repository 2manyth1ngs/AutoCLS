"""WP16 T 检验（转导型 E_T）——先于实现（铁律 5）。协议 results/tclsc_s1et/PROTOCOL.md。

A. `mech_features.rep_profile(encode_fn, X, y, task_type=)`：22 维 REP_PROFILE_COLS（rep_<op> 11 / rcos_<op> 10 / rinfo），恒等编码器下 rep_ 与
   原始空间 retention_info 同值、rcos 与 distortion_cos 同值；task_type 分发。
B. 33 EA_ARMS / 63 ARMS 的 ft60 / ft290 / ftea 臂与列表（`et_cols(budget)`）；63 的 et 表合并；69 号门与读数码、特征行结构。
零 GPU、不读真实缓存。
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from autocls.data.mech_features import RESPONSE_OPS, RESPONSE_PROFILE_COLS
from autocls.task_types import TASK_CLASSIFICATION
from scripts import pq_common as pq

m33 = importlib.import_module("scripts.33_pq_comparator_loo")
EA_COLS = tuple(RESPONSE_PROFILE_COLS)
PHI_COLS = tuple(m33.PHI_COLS)


def _mf():
    return importlib.import_module("autocls.data.mech_features")


def _g63():
    return importlib.import_module("scripts.63_tclsc_s1ea")


def _g69():
    return importlib.import_module("scripts.69_tclsc_s1et")


def _blobs(n_per=12, T=32, C=2, k=3, sep=6.0, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(k):
        center = rng.standard_normal((T, C)) * sep
        X.append(center[None] + rng.standard_normal((n_per, T, C)))
        y += [c] * n_per
    return np.concatenate(X).astype(np.float32), np.array(y)


def _identity_encode(X):
    return np.asarray(X, np.float32).reshape(len(X), -1)


class TestRepProfile:
    def test_cols(self):
        mf = _mf()
        cols = mf.REP_PROFILE_COLS
        assert len(cols) == 22 and len(set(cols)) == 22
        assert sum(c.startswith("rep_") for c in cols) == 11 and sum(c.startswith("rcos_") for c in cols) == 10 and "rinfo" in cols
        assert "rcos_polarity" not in cols and "rep_polarity" in cols
        assert mf.rep_cols_at(60) == tuple(f"{c}@60" for c in cols) and len(mf.rep_cols_at(290)) == 22

    def test_identity_encoder_matches_raw_profile(self):
        mf = _mf()
        X, y = _blobs()
        rep = mf.rep_profile(_identity_encode, X, y, task_type=TASK_CLASSIFICATION, seeds=(0, 1))
        raw = mf.response_profile(X, y, task_type=TASK_CLASSIFICATION, seeds=(0, 1))
        for op, spec in RESPONSE_OPS.items():
            assert rep[f"rep_{op}"] == pytest.approx(raw[f"ret_{op}"], abs=1e-6), op
            if spec["sim"]:
                assert rep[f"rcos_{op}"] == pytest.approx(raw[f"sim_{op}"], abs=1e-6), op
        assert rep["rinfo"] == pytest.approx(raw["info_raw"], abs=1e-6)
        assert all(np.isfinite(rep[c]) for c in mf.REP_PROFILE_COLS)
        with pytest.raises(NotImplementedError):
            mf.rep_profile(_identity_encode, X, y, task_type="forecasting")

    def test_torch_encoder_path(self):
        """真实路径：nn.Module (B,T,C)→(B,T,d) 经 encode_features 时间维 max-pool。"""
        mf = _mf()
        X, y = _blobs()
        enc = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.ReLU())
        fn = mf.torch_encode_fn(enc, device="cpu")
        Z = fn(X)
        assert Z.shape == (len(y), 8)
        rep = mf.rep_profile(fn, X, y, task_type=TASK_CLASSIFICATION, seeds=(0,))
        assert all(-1.0 <= rep[c] <= 1.0 for c in mf.REP_PROFILE_COLS)


class TestArmsAndCols:
    def test_33_and_63_arms(self):
        mf = _mf()
        a = m33.EA_ARMS
        assert a["ft60_hard"].head == "field" and a["ft60_hard"].ctx_kind == "table_budget" and a["ft60_hard"].head_rank is None
        assert a["ft60_hard"].ctx_cols == mf.rep_cols_at(60) + PHI_COLS
        assert a["ft290_hard"].ctx_cols == mf.rep_cols_at(290) + PHI_COLS
        assert a["ftea_hard"].ctx_cols == mf.rep_cols_at(60) + EA_COLS + PHI_COLS and len(a["ftea_hard"].ctx_cols) == 59
        g = _g63()
        assert g.ARMS["u_ft60_hard"].ctx_kind == "et60_budget" and g.ARMS["u_ftea_hard"].ctx_kind == "etea_budget" and g.ARMS["u_ft290_hard"].head == "field"
        assert g.ctx_cols("et60_budget") == mf.rep_cols_at(60) + PHI_COLS and g.ctx_cols("etea_budget") == mf.rep_cols_at(60) + EA_COLS + PHI_COLS

    def test_63_table_for_kind_merges_et(self):
        g = _g63()
        mf = _mf()
        ea = {"D": {"dataset": "D", **{c: 1.0 for c in EA_COLS}}}
        phi = {"D": {"dataset": "D", **{c: 2.0 for c in PHI_COLS}}}
        et = {"D": {"dataset": "D", **{c: 3.0 for c in mf.rep_cols_at(60) + mf.rep_cols_at(290)}}}
        t = g.table_for_kind("etea_budget", ea, phi, et)
        assert t["D"][EA_COLS[0]] == 1.0 and t["D"][PHI_COLS[0]] == 2.0 and t["D"][mf.rep_cols_at(60)[0]] == 3.0
        assert g.table_for_kind("eaphi_budget", ea, phi, et)["D"][PHI_COLS[0]] == 2.0
        with pytest.raises(ValueError):
            g.table_for_kind("et60_budget", ea, phi, None)


class TestScript69:
    def test_feature_row_and_gates(self):
        g = _g69()
        mf = _mf()
        row = g.feature_row("T_00", "T", {60: {c: 0.1 for c in mf.REP_PROFILE_COLS}, 290: {c: 0.2 for c in mf.REP_PROFILE_COLS}}, seconds=1.0, acc290=0.5)
        assert row["task_id"] == "T_00" and row[mf.rep_cols_at(60)[0]] == 0.1 and row[mf.rep_cols_at(290)[0]] == 0.2 and row["acc290_probe"] == 0.5
        v = lambda a, b, p: {"mean_a": a, "mean_b": b, "p": p}
        gt = g.gates_et(v(0.0269, 0.024, 0.01), v(0.0294, 0.024, 0.01), v(0.024, 0.028, 0.01))
        assert g.verdict_et(gt) == "ET-ALIVE+CONTENT"
        assert g.verdict_et(g.gates_et(v(0.0269, 0.027, 0.6), v(0.0294, 0.026, 0.02), v(0.026, 0.026, 0.5))) == "ET-EQUIV-CONTENT"
        assert g.verdict_et(g.gates_et(v(0.0269, 0.031, 0.9), v(0.0294, 0.031, 0.9), v(0.031, 0.034, 0.01))) == "ET-DEAD+CONTENT"
        assert g.UEA_ARMS == ("ft60_hard", "ft290_hard", "ftea_hard") and g.UCR_ARMS == ("u_ft60_hard", "u_ft290_hard", "u_ftea_hard")
        assert g.PROBE_BUDGETS == (60, 290)
