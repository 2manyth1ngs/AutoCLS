"""WP16-TCLSC 阶段 1 A′ 确认版——先于实现（铁律 5）。协议 results/tclsc_s1ea_confirm/PROTOCOL.md。

A. `scripts/33`：通用 `ctx_kind="table_budget"`（`ArmSpec.ctx_cols` + `PQContext(ctx_file=)`）——维度、向量、nan 感知折内标准化、
   `EA_ARMS` 表（ea_diff_hard 25 / eaphi_diff_hard 37）、冒烟目录训 `eaphi_diff_hard` → meta 记 ctx_kind / ctx_cols / ctx_mu / ctx_sd / n_nan。
B. `scripts/64_tclsc_s1ea_confirm.py`：切片键 ctx 行（normal / cross / zero 只置零 E_A 列）、UEA 切片特征行（含 φ 并入）、
   参照臂排名合并（s1phi 结构）、门 G-C1/2/3 与读数码 + 后缀、UCR 新种子端到端（60 号 + 63 号训练器同一目录）。
零 GPU、不读真实缓存。
"""
from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from autocls.data.mech_features import RESPONSE_PROFILE_COLS
from autocls.data.task_slicer import SlicedTask
from autocls.space.cls_space import CLSSpace
from scripts import pq_common as pq

m33 = importlib.import_module("scripts.33_pq_comparator_loo")
g34 = importlib.import_module("scripts.34_pq_search")
SPACE = CLSSpace.from_yaml()
EA_COLS = tuple(RESPONSE_PROFILE_COLS)
PHI_COLS = tuple(m33.PHI_COLS)
ALL_COLS = EA_COLS + PHI_COLS


def _g64():
    return importlib.import_module("scripts.64_tclsc_s1ea_confirm")


def _table(slices, cols=ALL_COLS, seed=0, nan_at=None):
    rng = np.random.default_rng(seed)
    t = {s: {"task_id": s, "dataset": pq.dataset_of(s), **{c: float(rng.standard_normal()) for c in cols}} for s in slices}
    if nan_at:
        s, c = nan_at
        t[s][c] = None
    return t


def _write(path, table):
    with open(path, "w", encoding="utf-8") as fh:
        for r in table.values():
            fh.write(json.dumps(r) + "\n")
    return path


# ================================================================ A. 33 号
class TestTableCtx33:
    def test_ea_arms_table(self):
        a = m33.EA_ARMS
        assert {"ea_diff_hard", "eaphi_diff_hard"} <= set(a)                    # R 检验又加了 fw4_ / fw_ 臂（test_tclsc_s1ea_rank）
        assert not (set(a) & (set(m33.ARMS) | set(m33.L3_ARMS) | set(m33.PHI_ARMS) | set(m33.FP_ARMS) | set(m33.E_ARMS)))
        for s in a.values():
            assert s.split == "S1" and s.seeds == (0, 1, 2) and s.head in ("diff", "field") and s.label_mode == "hard"
            assert s.ctx_kind == "table_budget" and s.task_branch == "ctx" and s.encoder is None
        assert a["ea_diff_hard"].head == "diff" and a["eaphi_diff_hard"].head == "diff"
        assert a["ea_diff_hard"].ctx_cols == EA_COLS and a["eaphi_diff_hard"].ctx_cols == ALL_COLS
        assert m33.arm_spec("eaphi_diff_hard") is a["eaphi_diff_hard"] and "eaphi_diff_hard" in m33.CLI_ARMS
        assert m33.ArmSpec("x", "S1", "ctx", None, None, "budget").ctx_cols is None

    def test_ctx_dim_table(self):
        assert m33.ctx_dim_for(m33.EA_ARMS["ea_diff_hard"], pq.BUDGETS, pq.PQ_DATASETS) == 25 + 3
        assert m33.ctx_dim_for(m33.EA_ARMS["eaphi_diff_hard"], pq.BUDGETS, pq.PQ_DATASETS) == 37 + 3

    def test_table_standardizer_nan_aware(self):
        slices = ["Cricket_00", "Cricket_01", "Handwriting_00"]
        t = _table(slices, nan_at=("Cricket_01", EA_COLS[0]))
        mu, sd = m33.table_standardizer(t, ["Cricket_00", "Cricket_01"], ALL_COLS)
        assert mu.shape == (37,) and np.all(sd >= m33.PHI_SD_FLOOR)
        assert mu[0] == pytest.approx(t["Cricket_00"][EA_COLS[0]])
        assert mu[1] == pytest.approx((t["Cricket_00"][EA_COLS[1]] + t["Cricket_01"][EA_COLS[1]]) / 2)

    def test_context_for_table(self):
        spec = m33.EA_ARMS["eaphi_diff_hard"]
        slices = ["Cricket_00", "Cricket_01", "Handwriting_00"]
        t = _table(slices, nan_at=("Handwriting_00", EA_COLS[3]))
        mu, sd = m33.table_standardizer(t, ["Cricket_00", "Cricket_01"], ALL_COLS)
        tids = [pq.engine_task_id(s, b) for s in slices for b in pq.BUDGETS]
        ctx = m33.context_for(spec, tids, pq.BUDGETS, pq.PQ_DATASETS, table=t, table_mu=mu, table_sd=sd)
        v = ctx[pq.engine_task_id("Handwriting_00", 120)]
        assert v.shape == (40,) and v.dtype == np.float32 and v[3] == 0.0
        assert v[37:].tolist() == pq.ctx_vector(120, pq.BUDGETS).tolist()
        x = np.array([t["Cricket_00"][c] for c in ALL_COLS])
        assert np.allclose(ctx[pq.engine_task_id("Cricket_00", 60)][:37], (x - mu) / sd, atol=1e-6)
        with pytest.raises(ValueError):
            m33.context_for(spec, tids, pq.BUDGETS, pq.PQ_DATASETS)

    def test_train_eaphi_on_smoke_dir(self, tmp_path):
        m33.synth_smoke_inputs(tmp_path, datasets=("Handwriting", "Cricket"), n_slices=3, n_cls=4, budgets=pq.BUDGETS)
        slices = [f"{d}_{i:02d}" for d in ("Handwriting", "Cricket") for i in range(3)]
        ctx_file = _write(tmp_path / "ea.jsonl", _table(slices, nan_at=("Cricket_01", EA_COLS[7])))
        m33.cmd_train(tmp_path, arms=["eaphi_diff_hard"], folds=["Handwriting"], seeds=[0], resume=False, device="cpu",
                      epochs=2, settf_windows=2, smoke=True, encoder_mode="binned", ctx_file=ctx_file)
        meta = torch.load(tmp_path / "models" / "Handwriting" / "eaphi_diff_hard_s0.pt", map_location="cpu", weights_only=True)["meta"]
        assert meta["head"] == "diff" and meta["ctx_dim"] == 40 and meta["ctx_kind"] == "table_budget"
        assert list(meta["ctx_cols"]) == list(ALL_COLS) and len(meta["ctx_mu"]) == 37 and meta["n_nan"] == {"Cricket_01": 1}
        t = {r["task_id"]: r for r in map(json.loads, open(ctx_file, encoding="utf-8"))}
        mu, sd = m33.table_standardizer(t, [s for s in slices if s.startswith("Cricket")], ALL_COLS)
        assert np.allclose(meta["ctx_mu"], mu) and np.allclose(meta["ctx_sd"], sd)
        run = json.loads((tmp_path / "runs" / "Handwriting" / "eaphi_diff_hard_s0.json").read_text(encoding="utf-8"))
        assert run["settings"]["ctx_kind"] == "table_budget" and run["settings"]["ctx_dim"] == 40 and run["cross"] is not None
        model, _ = g34.load_pq_model(tmp_path / "models" / "Handwriting" / "eaphi_diff_hard_s0.pt", expect_fold="Handwriting", expect_budgets=pq.BUDGETS)
        assert model.head == "diff" and model.ctx_dim == 40

    def test_train_table_arm_without_file_fails_fast(self, tmp_path):
        m33.synth_smoke_inputs(tmp_path, datasets=("Handwriting", "Cricket"), n_slices=3, n_cls=4, budgets=pq.BUDGETS)
        with pytest.raises((FileNotFoundError, ValueError, SystemExit)):
            m33.cmd_train(tmp_path, arms=["ea_diff_hard"], folds=["Handwriting"], seeds=[0], resume=False, device="cpu",
                          epochs=2, settf_windows=2, smoke=True, encoder_mode="binned", ctx_file=None)


# ================================================================ B. 64 号
class TestSliceCtxRows:
    def test_rows_cross_zero_keep_phi(self):
        g = _g64()
        slices = ["Handwriting_04", "Cricket_00", "Cricket_01", "UWaveGestureLibrary_00"]
        t = _table(slices)
        mu, sd = m33.table_standardizer(t, slices[1:], ALL_COLS)
        rows = g.slice_ctx_rows(t, ALL_COLS, ["Handwriting_04"], mu, sd)
        assert set(rows) == {("Handwriting_04", b) for b in pq.BUDGETS} and rows[("Handwriting_04", 60)].shape == (40,)
        z = g.slice_ctx_rows(t, ALL_COLS, ["Handwriting_04"], mu, sd, zero_cols=EA_COLS)
        assert np.all(z[("Handwriting_04", 60)][:25] == 0) and np.allclose(z[("Handwriting_04", 60)][25:37], rows[("Handwriting_04", 60)][25:37])
        x = g.slice_ctx_rows(t, ALL_COLS, ["Handwriting_04"], mu, sd, cross_map={"Handwriting_04": "Cricket_01"})
        own = g.slice_ctx_rows(t, ALL_COLS, ["Cricket_01"], mu, sd)
        assert np.allclose(x[("Handwriting_04", 290)][:37], own[("Cricket_01", 290)][:37])


class TestFeaturesUea:
    def test_rows_from_dataset_objects(self):
        g = _g64()
        rng = np.random.default_rng(0)
        X = np.concatenate([rng.standard_normal((20, 32, 2)) + 5 * c for c in range(3)]).astype(np.float32)
        y = np.repeat(np.arange(3), 20)
        t = SlicedTask(task_id="DSA_00", pool="DSA", classes=(0, 1, 2), channels=(0, 1), sample_indices=tuple(range(60)), tag="", task_type="classification")
        ds = {"DSA": SimpleNamespace(slices=[t], X=X, y=y)}
        phi = {"DSA_00": {"task_id": "DSA_00", **{c: 1.0 for c in PHI_COLS}}}
        rows = list(g.ea_rows_uea(ds, phi, done=set(), seeds=(0,)))
        assert len(rows) == 1 and rows[0]["task_id"] == "DSA_00" and rows[0]["dataset"] == "DSA"
        assert all(c in rows[0] for c in ALL_COLS) and rows[0]["n_train"] == 42 and rows[0][PHI_COLS[0]] == 1.0
        assert list(g.ea_rows_uea(ds, phi, done={"DSA_00"}, seeds=(0,))) == []


class TestMergeAndGates:
    def test_merge_ref_by_target(self):
        g = _g64()
        own = {"Cricket": {"normal": {"60": {"Cricket_00": {"eaphi_diff_hard": ["a", "b"]}}}, "cross": {}, "zero": {}}}
        ref = {"Cricket": {"normal": {"60": {"Cricket_00": {"blind_diff_hard": ["b", "a"], "phi_diff_hard": ["a", "b"], "phi_concat_hard": ["b", "a"]}}},
                           "cross": {}, "zero": {}}}
        m = g.merge_ref(own, ref, arms=g.REF_ARMS)
        assert set(m["Cricket"]["normal"]["60"]["Cricket_00"]) == {"eaphi_diff_hard", "blind_diff_hard", "phi_diff_hard"}
        with pytest.raises(ValueError):
            g.merge_ref(own, {}, arms=g.REF_ARMS)

    def test_gates_and_codes(self):
        g = _g64()
        v = lambda a, b, p: {"mean_a": a, "mean_b": b, "p": p}
        gc = g.gates_confirm(v(0.031, 0.027, 0.01), v(0.027, 0.031, 0.01), v(0.0209, 0.0200, 0.01))
        assert gc["GC1"]["pass"] and gc["GC2"]["pass"] and gc["GC3"]["pass"]
        assert g.verdict_confirm(gc) == "EA-CONFIRMED+UCR-REP"
        assert g.verdict_confirm(g.gates_confirm(v(0.031, 0.027, 0.01), v(0.027, 0.028, 0.4), v(0.0209, 0.0200, 0.5))) == "EA-VALUE-NO-CONTENT-UCR-REP"
        assert g.verdict_confirm(g.gates_confirm(v(0.031, 0.030, 0.3), v(0.030, 0.034, 0.01), v(0.0209, 0.0200, 0.01))) == "EA-CONTENT-NO-VALUE+UCR-REP"
        assert g.verdict_confirm(g.gates_confirm(v(0.031, 0.032, 0.9), v(0.032, 0.032, 0.5), v(0.0209, 0.0210, 0.5))) == "EA-NOT-CONFIRMED-UCR-REP"
        # G-C2 方向：cross 必须**更差**（mean_b > mean_a）才算内容依赖
        assert not g.gates_confirm(v(0.031, 0.027, 0.01), v(0.030, 0.027, 0.01), v(0.02, 0.02, 0.5))["GC2"]["pass"]
        assert g.verdict_confirm(g.gates_confirm(v(0.031, 0.027, 0.01), v(0.027, 0.031, 0.01), {"mean_a": None, "mean_b": None, "p": None})) == "EA-CONFIRMED?UCR-REP"


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


class TestUcrReplication:
    def test_train_and_rank_three_arms_one_dir(self, tmp_path):
        g = _g64()
        rows = _ucr_rows()
        ds = sorted({r["dataset"] for r in rows})
        rng = np.random.default_rng(1)
        ea = {d: {"dataset": d, "task_id": f"{d}_00", **{c: float(rng.standard_normal()) for c in EA_COLS}} for d in ds}
        phi = {d: {"task_id": f"{d}_00", "dataset": d, **{c: float(rng.standard_normal()) for c in PHI_COLS}} for d in ds}
        folds = [["DS02", "DS03"], ["DS00", "DS01"]]
        out = tmp_path / "ucr"
        g.train_ucr(rows, ea, phi, folds, out, seeds=[3], folds_sel=[0], device="cpu", epochs=2, resume=False)
        for a in g.UCR_ARMS:
            assert (out / "models" / "fold0" / f"{a}_s3.pt").exists()
        rk = g.rank_ucr(rows, ea, phi, folds, out, seeds=[3], folds_sel=[0])
        per = rk["fold0"]["normal"]["60"]["DS02_00"]
        assert set(per) == set(g.UCR_ARMS)
        assert "u_eaphi_diff_hard" in rk["fold0"]["zero"]["60"]["DS02_00"] and set(rk["fold0"]["cross"]) == {"0", "1", "2"}
        assert g.UCR_ARMS == ("u_blind_diff_hard", "u_phi_diff_hard", "u_eaphi_diff_hard")
