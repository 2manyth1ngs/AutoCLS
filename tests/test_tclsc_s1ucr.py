"""WP16-TCLSC 阶段 1 n 放大（UCR 84）——先于实现（铁律 5）。协议 results/tclsc_s1ucr/PROTOCOL.md。

覆盖 `scripts/60_tclsc_s1ucr.py`：分组 7 折（逐位确定、互斥、每折 12）；记录构建（engine tid、binned 116、origin）；
目录（每集 20 条）；ctx 表（φ / 指纹，折内标准化；任务盲 None）；δ 常数；门与读数码；端到端小样本训练 → meta / 排名。
零 GPU、不读真实缓存。
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from autocls.space.cls_space import CLSSpace
from scripts import pq_common as pq

m33 = importlib.import_module("scripts.33_pq_comparator_loo")
SPACE = CLSSpace.from_yaml()


def _g60():
    return importlib.import_module("scripts.60_tclsc_s1ucr")


def _ucr_rows(n_datasets=14, n_cls=20, seed=0):
    """合成 UCR 形状的缓存行：<name>_00，20 条 CLS（前 10 shared，后 10 random），三截点。"""
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


def _phi(rows):
    rng = np.random.default_rng(1)
    return {d: {"task_id": f"{d}_00", "dataset": d, **{c: float(rng.standard_normal()) for c in m33.PHI_COLS}}
            for d in sorted({r["dataset"] for r in rows})}


class TestFolds:
    def test_seven_grouped_folds_deterministic(self):
        g = _g60()
        names = [f"DS{i:02d}" for i in range(84)]
        folds = g.grouped_folds(names, k=7)
        assert len(folds) == 7 and all(len(f) == 12 for f in folds)
        assert sorted(x for f in folds for x in f) == sorted(names)
        assert folds == g.grouped_folds(list(reversed(names)), k=7)       # 与输入顺序无关（内部排序）

    def test_delta_constant(self):
        assert _g60().DELTA_UCR == 0.06


class TestRecords:
    def test_records_and_catalog(self):
        g = _g60()
        rows = _ucr_rows(n_datasets=3)
        recs = g.build_records(rows, ["DS00", "DS01"])
        assert len(recs) == 2 * 20 * 3
        assert all(r.vec.shape == (116,) for r in recs)
        assert {pq.split_engine_task_id(r.task_id)[0] for r in recs} == {"DS00_00", "DS01_00"}
        assert {r.origin for r in recs} == {"shared", "random"}
        cat = g.catalog_for(rows, "DS02")
        assert set(cat) == {"DS02_00"} and len(cat["DS02_00"]) == 20
        assert sum(1 for o, _c, _k in cat["DS02_00"] if o == "shared") == 10


class TestCtx:
    def test_phi_ctx_uses_train_stats_only(self):
        g = _g60()
        rows = _ucr_rows(n_datasets=4)
        phi = _phi(rows)
        train, test = ["DS00", "DS01", "DS02"], ["DS03"]
        ctx_map, meta = g.phi_context(phi, train, test)
        tid = pq.engine_task_id("DS03_00", 120)
        assert ctx_map[tid].shape == (12 + 3,)
        mu, sd = m33.phi_standardizer({f"{d}_00": phi[d] for d in phi}, [f"{d}_00" for d in train], m33.PHI_COLS)
        assert np.allclose(meta["phi_mu"], mu) and np.allclose(meta["phi_sd"], sd)
        assert np.allclose(ctx_map[tid][:12], (np.array([phi["DS03"][c] for c in m33.PHI_COLS]) - mu) / sd)

    def test_fp_ctx_probes_from_train_only(self):
        g = _g60()
        rows = _ucr_rows(n_datasets=4)
        cache = g.rows_to_cache(rows)
        train, test = ["DS00", "DS01", "DS02"], ["DS03"]
        ctx_map, meta = g.fp_context(cache, train, test)
        tid = pq.engine_task_id("DS03_00", 60)
        assert ctx_map[tid].shape == (m33.FP_DIM + 3,)
        assert len(meta["fp_probes"]) == m33.FP_K
        shared = {r["cls_key"] for r in rows if r["origin"] == "shared"}
        assert set(meta["fp_probes"]) <= shared
        assert meta["fp_probes"] == m33.fp_probes(cache, [f"{d}_00" for d in train], k=m33.FP_K)

    def test_blind_ctx_is_none(self):
        g = _g60()
        assert g.ARMS["u_blind_diff_hard"].ctx_kind == "none" and g.ARMS["u_blind_diff_hard"].head == "diff"
        assert g.ARMS["u_phi_concat_hard"].head == "concat" and g.ARMS["u_phi_diff_hard"].ctx_kind == "phi_budget"
        assert g.ARMS["u_fp_diff_hard"].ctx_kind == "fp_budget"
        assert all(s.split == "S1" and s.seeds == (0, 1, 2) for s in g.ARMS.values())


class TestEndToEnd:
    def test_train_one_fold_writes_meta_and_ranks(self, tmp_path):
        """4 数据集、2 训 2 考、epochs=2：模型 meta 记 ctx_kind / 统计，排名给出全部 20 键的排列。"""
        g = _g60()
        rows = _ucr_rows(n_datasets=4)
        phi = _phi(rows)
        outdir = tmp_path / "ucr"
        folds = [["DS02", "DS03"], ["DS00", "DS01"]]
        g.train_fold(rows, phi, folds, 0, arm="u_fp_diff_hard", seed=0, outdir=outdir, device="cpu", epochs=2)
        g.train_fold(rows, phi, folds, 0, arm="u_phi_diff_hard", seed=0, outdir=outdir, device="cpu", epochs=2)
        import torch
        meta = torch.load(outdir / "models" / "fold0" / "u_fp_diff_hard_s0.pt", map_location="cpu", weights_only=True)["meta"]
        assert meta["ctx_kind"] == "fp_budget" and meta["fold"] == "fold0" and meta["head"] == "diff"
        meta2 = torch.load(outdir / "models" / "fold0" / "u_phi_diff_hard_s0.pt", map_location="cpu", weights_only=True)["meta"]
        assert meta2["ctx_kind"] == "phi_budget" and len(meta2["phi_mu"]) == 12
        rk = g.rank_fold(rows, phi, folds, 0, arms=["u_fp_diff_hard", "u_phi_diff_hard"], seeds=[0], outdir=outdir)
        for name in ("DS02_00", "DS03_00"):
            for b in pq.BUDGETS:
                got = rk["normal"][str(b)][name]
                keys = sorted({r["cls_key"] for r in rows})
                assert sorted(got["u_fp_diff_hard"]) == keys and sorted(got["u_phi_diff_hard"]) == keys
        assert set(rk["cross"]) == {"0", "1", "2"} and rk["zero"] and rk["probes"]


class TestGatesUcr:
    def _phi_inputs(self, d, p, reg_phi, reg_blind, p2):
        return {"u_phi_diff_hard": {"acc_top10": {"mean_delta": d, "p": p}},
                "u_phi_concat_hard": {"acc_top10": {"mean_delta": 0.0, "p": 0.9}}}, {"mean_a": reg_blind, "mean_b": reg_phi, "p": p2}

    def test_phi_codes(self):
        g = _g60()
        assert g.verdict_phi(g.gates_phi(*self._phi_inputs(0.05, 0.01, 0.018, 0.020, 0.01))) == "UCR-PHI-ALIVE"
        assert g.verdict_phi(g.gates_phi(*self._phi_inputs(0.05, 0.01, 0.021, 0.020, 0.6))) == "UCR-PHI-USED-NO-GAIN"
        assert g.verdict_phi(g.gates_phi(*self._phi_inputs(0.01, 0.01, 0.018, 0.020, 0.01))) == "UCR-PHI-DEAD"

    def test_fp_codes(self):
        g = _g60()
        d = {"u_fp_diff_hard": {"acc_top10": {"mean_delta": 0.05, "p": 0.01}}}
        v2 = {"mean_a": 0.020, "mean_b": 0.018, "p": 0.01}
        v3 = {"mean_a": 0.015, "mean_b": 0.014, "p": 0.01}
        assert g.verdict_fp(g.gates_fp(d, v2, v3)) == "UCR-FP-ALIVE"
        assert g.verdict_fp(g.gates_fp(d, v2, dict(v3, p=0.5))) == "UCR-FP-EXTRAPOLATES"
        assert g.verdict_fp(g.gates_fp(d, dict(v2, p=0.5), v3)) == "UCR-FP-USED-NO-GAIN"
        assert g.verdict_fp(g.gates_fp({"u_fp_diff_hard": {"acc_top10": {"mean_delta": 0.0, "p": 0.5}}}, v2, v3)) == "UCR-FP-DEAD"
