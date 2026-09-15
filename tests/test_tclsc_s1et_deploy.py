"""D 检验（官方口径零样本名单模拟）——先于实现。覆盖：官方目录轨迹表构造、SH 模拟一条名单（36 号同式）、ctx 构造（E_T@60 ⊕ φ ⊕ 预算 290）、门与读数码。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from autocls.data.mech_features import REP_PROFILE_COLS, rep_cols_at
from scripts import pq_common as pq

m33 = importlib.import_module("scripts.33_pq_comparator_loo")
PHI_COLS = tuple(m33.PHI_COLS)


def _g71():
    return importlib.import_module("scripts.71_tclsc_s1et_deploy")


def _runs():
    rows = []
    for i, k in enumerate("abcdef"):
        accs = [0.30 + 0.05 * i, 0.35 + 0.05 * i, 0.40 + 0.05 * i]
        rows.append({"dataset": "T", "stage": "s1_catalog", "cls_key": k, "seed": 0, "iters": 200, "checkpoints": [50, 100, 200],
                     "arm": "shared", "cls": {}, "acc": accs[-1],
                     "checkpoint_results": [{"iters": b, "acc": a} for b, a in zip((50, 100, 200), accs)]})
    rows.append({"dataset": "T", "stage": "s2_finals", "cls_key": "f", "seed": 1, "iters": 200, "acc": 0.64, "checkpoints": [], "checkpoint_results": []})
    return rows


class TestTrajAndSH:
    def test_traj_and_finals(self):
        g = _g71()
        traj, finals, budgets = g.catalog_traj(_runs(), "T")
        assert budgets == (50, 100, 200) and set(traj) == set("abcdef") and traj["f"][200] == pytest.approx(0.65)
        assert finals["f"] == [0.65, 0.64]                                  # seed 0 在前 + s2_finals 种子
        assert finals["a"] == [0.40]                                          # 只有 seed 0

    def test_sh_on_list(self):
        g = _g71()
        traj, _f, budgets = g.catalog_traj(_runs(), "T")
        r = g.sh_on_list(["a", "b", "c", "d", "e", "f"], traj, ks=(6, 3, 3), budgets=budgets)
        assert r["champion"] == "f" and r["acc"] == pytest.approx(0.65) and r["cost_iters"] == 6 * 50 + 3 * 50 + 3 * 100
        r2 = g.sh_on_list(["a", "b", "c", "d"], traj, ks=(6, 3, 3), budgets=budgets)           # 名单不足 6 条：按实际数
        assert r2["champion"] == "d"
        assert g.SH_ARMS == {"SH6-3-3": (6, 3, 3), "SH10-4-3": (10, 4, 3)}


class TestCtx:
    def test_official_ctx_rows(self):
        g = _g71()
        cols = rep_cols_at(60) + PHI_COLS
        mu, sd = np.zeros(len(cols)), np.ones(len(cols))
        et = {c: 0.5 for c in REP_PROFILE_COLS}
        phi = {c: 2.0 for c in PHI_COLS}
        rows = g.official_ctx_rows("T", et, phi, cols, mu, sd)
        assert set(rows) == {("T_official", b) for b in pq.BUDGETS}
        v = rows[("T_official", 290)]
        assert v.shape == (len(cols) + 3,) and v[0] == 0.5 and v[22] == 2.0 and v[-3:].tolist() == pq.ctx_vector(290, pq.BUDGETS).tolist()
        assert g.DEPLOY_BUDGET == 290


class TestGates:
    def test_codes(self):
        g = _g71()
        assert g.verdict_deploy(g.gates_deploy(0.735, 0.719, 5, 5)) == "DEPLOY-ET-WIN"
        assert g.verdict_deploy(g.gates_deploy(0.735, 0.719, 5, 4)) == "DEPLOY-ET-BETTER"
        assert g.verdict_deploy(g.gates_deploy(0.722, 0.719, 5, 6)) == "DEPLOY-ET-FLAT"        # 均值差 < 0.005
        assert g.verdict_deploy(g.gates_deploy(0.735, 0.719, 3, 6)) == "DEPLOY-ET-FLAT"        # 不低于先验的靶数 < 4
