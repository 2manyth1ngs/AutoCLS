"""WP16-TCLSC 阶段 1 A″ 跨语料部署检验——先于实现（铁律 5）。协议 results/tclsc_s1ea_xfer/PROTOCOL.md。

覆盖 `scripts/65_tclsc_s1ea_xfer.py`：无折全训（train = 全部数据集、test = ∅，三臂两训练器同目录）、跨语料护栏（死列置 0、裁剪 ±4）的
切片键 ctx 行（normal / cross / zero）、单靶排名（三臂 + 消融）、参照臂合并、门 G-X1/2/3 与读数码 + 后缀。零 GPU、不读真实缓存。
"""
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


def _g65():
    return importlib.import_module("scripts.65_tclsc_s1ea_xfer")


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
    return rows, cands


def _tables(ds, seed=1):
    rng = np.random.default_rng(seed)
    ea = {d: {"dataset": d, "task_id": f"{d}_00", **{c: float(rng.standard_normal()) for c in EA_COLS}} for d in ds}
    phi = {d: {"task_id": f"{d}_00", "dataset": d, **{c: float(rng.standard_normal()) for c in PHI_COLS}} for d in ds}
    for d in ds:                       # 模拟 UCR 单变量：log10_C / chan_cv 恒 0
        phi[d]["log10_C"] = 0.0
        phi[d]["chan_cv"] = 0.0
    return ea, phi


class TestCtxGuard:
    def test_dead_cols_and_clip(self):
        g = _g65()
        sd = np.array([1.0, m33.PHI_SD_FLOOR, 5 * m33.PHI_SD_FLOOR, 0.5])
        assert g.dead_cols(sd).tolist() == [False, True, True, False]
        cols = ("a", "b", "c", "d")
        table = {"T_00": {"a": 100.0, "b": 0.7, "c": 0.0, "d": None}, "U_00": {"a": 1.0, "b": 0.0, "c": 0.0, "d": 2.0}}
        mu = np.zeros(4)
        rows = g.xfer_ctx_rows(table, cols, ["T_00"], mu, sd)
        v = rows[("T_00", 60)]
        assert v.shape == (7,) and v[0] == 4.0 and v[1] == 0.0 and v[2] == 0.0 and v[3] == 0.0       # 裁剪 / 死列 / nan → 0
        assert v[4:].tolist() == pq.ctx_vector(60, pq.BUDGETS).tolist()
        z = g.xfer_ctx_rows(table, cols, ["T_00"], mu, sd, zero_cols=("a",))
        assert z[("T_00", 60)][0] == 0.0
        x = g.xfer_ctx_rows(table, cols, ["T_00"], mu, sd, cross_map={"T_00": "U_00"})
        assert x[("T_00", 120)][0] == 1.0 and x[("T_00", 120)][3] == 2.0
        assert g.CLIP == 4.0


class TestTrainAll:
    def test_three_arms_no_folds(self, tmp_path):
        g = _g65()
        rows, _c = _ucr_rows()
        ds = sorted({r["dataset"] for r in rows})
        ea, phi = _tables(ds)
        g.train_all(rows, ea, phi, tmp_path, seeds=[0], device="cpu", epochs=2, resume=False)
        for a in g.ARMS:
            p = tmp_path / "models" / "fold0" / f"{a}_s0.pt"
            assert p.exists(), a
        meta = torch.load(tmp_path / "models" / "fold0" / "u_eaphi_diff_hard_s0.pt", map_location="cpu", weights_only=True)["meta"]
        assert meta["ctx_kind"] == "eaphi_budget" and len(meta["ctx_mu"]) == 37
        mu, _sd = importlib.import_module("scripts.63_tclsc_s1ea").table_standardizer(
            importlib.import_module("scripts.63_tclsc_s1ea").merge_tables(ea, phi), ds, EA_COLS + PHI_COLS)
        assert np.allclose(meta["ctx_mu"], mu)                                                      # 统计 = 全部数据集
        run = json.loads((tmp_path / "runs" / "fold0" / "u_phi_diff_hard_s0.json").read_text(encoding="utf-8"))
        assert run["test_datasets"] == [] and sorted(run["train_datasets"]) == ds
        assert g.ARMS == ("u_blind_diff_hard", "u_phi_diff_hard", "u_eaphi_diff_hard")


class TestRankTarget:
    def test_rank_target_arms_and_ablations(self, tmp_path):
        g = _g65()
        rows, cands = _ucr_rows()
        ds = sorted({r["dataset"] for r in rows})
        ea, phi = _tables(ds)
        g.train_all(rows, ea, phi, tmp_path, seeds=[0], device="cpu", epochs=2, resume=False)
        models = g.load_models(tmp_path, seeds=[0], device="cpu")
        assert set(models) == set(g.ARMS)
        # 目标侧：两个"UEA"数据集各 2 切片，候选 = 同 20 条
        from autocls.labeling import cls_key
        slices = ["TA_00", "TA_01", "TB_00", "TB_01"]
        cat = {s: [("shared" if j < 10 else "random", c, cls_key(c)) for j, c in enumerate(cands)] for s in slices}
        rng = np.random.default_rng(3)
        table = {s: {"task_id": s, "dataset": pq.dataset_of(s), **{c: float(rng.standard_normal()) for c in EA_COLS + PHI_COLS}} for s in slices}
        per = g.rank_target(models, cat, table, ["TA_00", "TA_01"], donor_slices=slices, device="cpu")
        keys = sorted(cls_key(c) for c in cands)
        for b in pq.BUDGETS:
            for s in ("TA_00", "TA_01"):
                got = per["normal"][str(b)][s]
                assert set(got) == set(g.ARMS) and all(sorted(v) == keys for v in got.values())
                assert set(per["zero"][str(b)][s]) == {"u_phi_diff_hard", "u_eaphi_diff_hard"}
        assert set(per["cross"]) == {"0", "1", "2"}
        for cs in per["cross"].values():
            assert all(pq.dataset_of(v) != pq.dataset_of(k) for k, v in cs["map"].items())
            assert set(cs["rankings"]["60"]["TA_00"]) == {"u_phi_diff_hard", "u_eaphi_diff_hard"}


class TestMergeGates:
    def test_merge_refs(self):
        g = _g65()
        own = {"Cricket": {"normal": {"60": {"Cricket_00": {"u_eaphi_diff_hard": ["a", "b"]}}}, "cross": {}, "zero": {}}}
        s1phi = {"Cricket": {"normal": {"60": {"Cricket_00": {"blind_diff_hard": ["b", "a"], "phi_diff_hard": ["a", "b"], "phi_concat_hard": ["a", "b"]}}}}}
        conf = {"Cricket": {"normal": {"60": {"Cricket_00": {"eaphi_diff_hard": ["b", "a"], "ea_diff_hard": ["a", "b"]}}}}}
        m = g.merge_refs(own, [(s1phi, ("blind_diff_hard", "phi_diff_hard")), (conf, ("eaphi_diff_hard",))])
        assert set(m["Cricket"]["normal"]["60"]["Cricket_00"]) == {"u_eaphi_diff_hard", "blind_diff_hard", "phi_diff_hard", "eaphi_diff_hard"}
        assert g.REF_ARMS == ("blind_diff_hard", "phi_diff_hard", "eaphi_diff_hard")

    def test_gates_codes(self):
        g = _g65()
        v = lambda a, b, p: {"mean_a": a, "mean_b": b, "p": p}
        gt = g.gates_xfer(v(0.0294, 0.025, 0.01), v(0.030, 0.025, 0.01), v(0.025, 0.030, 0.01))
        assert gt["GX1"]["pass"] and gt["GX2"]["pass"] and gt["GX3"]["pass"] and g.verdict_xfer(gt) == "XFER-ALIVE+CONTENT"
        assert g.verdict_xfer(g.gates_xfer(v(0.0294, 0.025, 0.01), v(0.025, 0.025, 0.6), v(0.025, 0.025, 0.5))) == "XFER-CORPUS-CONTENT"
        assert g.verdict_xfer(g.gates_xfer(v(0.0294, 0.035, 0.9), v(0.040, 0.035, 0.01), v(0.035, 0.040, 0.01))) == "XFER-RELATIVE+CONTENT"
        assert g.verdict_xfer(g.gates_xfer(v(0.0294, 0.035, 0.9), v(0.034, 0.035, 0.7), v(0.035, 0.034, 0.9))) == "XFER-DEAD-CONTENT"
        # G-X3 方向：cross 必须更差
        assert not g.gates_xfer(v(0.03, 0.02, 0.01), v(0.03, 0.02, 0.01), v(0.03, 0.02, 0.01))["GX3"]["pass"]
