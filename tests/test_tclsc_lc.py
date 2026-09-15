"""LC 检验（标签学习曲线）——先于实现。覆盖：三种训练侧子采样（配置 / 切片 / 数据集）的确定性与形状、UEA 折内训练（子采样 + ctx 标准化只用训练切片）、
判读规则（RISING / FLAT）与读数码。零 GPU、不读真实缓存。"""
from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from autocls.labeling import ORIGIN_RANDOM, ORIGIN_SHARED
from scripts import pq_common as pq

m33 = importlib.import_module("scripts.33_pq_comparator_loo")


def _g73():
    return importlib.import_module("scripts.73_tclsc_lc")


def _rows(datasets=("A", "B", "C"), n_slices=3, n_shared=10, n_random=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in datasets:
        for i in range(n_slices):
            s = f"{d}_{i:02d}"
            for j in range(n_shared):
                for b in pq.BUDGETS:
                    rows.append({"dataset": d, "task_id": s, "iters": int(b), "cls_key": f"S{j}", "origin": ORIGIN_SHARED, "r_prime": float(rng.random())})
            for j in range(n_random):
                for b in pq.BUDGETS:
                    rows.append({"dataset": d, "task_id": s, "iters": int(b), "cls_key": f"R{s}_{j}", "origin": ORIGIN_RANDOM, "r_prime": float(rng.random())})
    return rows


class TestSubsample:
    def test_config_axis_keeps_shared_paired(self):
        g = _g73()
        rows = _rows()
        sub = g.subsample_configs(rows, k=10, seed=0)
        by = {}
        for r in sub:
            by.setdefault(r["task_id"], set()).add(r["cls_key"])
        assert all(len(v) == 10 for v in by.values())
        shared_sets = {frozenset(k for k in v if k.startswith("S")) for v in by.values()}
        assert len(shared_sets) == 1 and len(next(iter(shared_sets))) == 5           # 全局同一批 shared
        assert all(sum(1 for k in v if not k.startswith("S")) == 5 for v in by.values())
        assert g.subsample_configs(rows, k=10, seed=0) == sub                       # 确定性
        full = g.subsample_configs(rows, k=20, seed=0)
        assert len(full) == len(rows)
        assert g.CONFIG_POINTS == (6, 10, 14, 20)

    def test_slice_axis(self):
        g = _g73()
        rows = _rows(n_slices=3)
        sub = g.subsample_slices(rows, s=2, seed=0)
        per = {}
        for r in sub:
            per.setdefault(r["dataset"], set()).add(r["task_id"])
        assert all(len(v) == 2 for v in per.values())
        assert g.subsample_slices(rows, s=2, seed=0) == sub
        assert g.SLICE_POINTS == (2, 4, 6)

    def test_dataset_axis(self):
        g = _g73()
        train = [f"D{i:02d}" for i in range(72)]
        sub = g.subsample_datasets(train, n=7, seed=0)
        assert len(sub) == 7 and set(sub) <= set(train) and sub == g.subsample_datasets(train, n=7, seed=0)
        assert g.subsample_datasets(train, n=72, seed=0) == sorted(train)
        assert g.DATASET_POINTS == (7, 14, 28, 56, 72)


class TestVerdict:
    def test_rising_flat_and_codes(self):
        g = _g73()
        # 点按大小升序：(size, e_regret3, blind_regret3)
        rising = [(6, 0.030, 0.030), (10, 0.026, 0.029), (14, 0.024, 0.029), (20, 0.021, 0.029)]
        flat = [(6, 0.030, 0.030), (10, 0.026, 0.029), (14, 0.0215, 0.029), (20, 0.021, 0.029)]
        assert g.axis_verdict(rising) == "RISING" and g.axis_verdict(flat) == "FLAT"
        # 涨了但 E 增益没涨（blind 涨得一样多）→ FLAT
        both = [(14, 0.024, 0.030), (20, 0.020, 0.026)]
        assert g.axis_verdict(both) == "FLAT"
        assert g.verdict_lc({"config": "RISING", "slice": "FLAT", "dataset": "RISING"}) == "LC-CONFIG+DATASET"
        assert g.verdict_lc({"config": "FLAT", "slice": "FLAT", "dataset": "FLAT"}) == "LC-SATURATED"
        assert g.verdict_lc({"config": "FLAT", "slice": "RISING", "dataset": "FLAT"}) == "LC-SLICE"


class TestTrainUea:
    def test_train_fold_uea_subsampled(self, tmp_path):
        g = _g73()
        m33.synth_smoke_inputs(tmp_path, datasets=("Handwriting", "Cricket"), n_slices=3, n_cls=6, budgets=pq.BUDGETS)
        rows, delta = g._g57.load_source(tmp_path)
        slices = sorted({r["task_id"] for r in rows})
        rng = np.random.default_rng(1)
        table = {s: {"task_id": s, "dataset": pq.dataset_of(s), **{c: float(rng.standard_normal()) for c in g.ET_COLS}} for s in slices}
        sub = g.subsample_slices(rows, s=2, seed=0)
        run = g.train_fold_uea(sub, delta, table, target="Handwriting", arm="ft60_hard", seed=0, outdir=tmp_path / "x", device="cpu", epochs=2, tag="S2")
        p = tmp_path / "x" / "models" / "S2" / "Handwriting" / "ft60_hard_s0.pt"
        assert p.exists() and run["n_train_slices"] == 2 and run["test"] == "Handwriting"
        meta = torch.load(p, map_location="cpu", weights_only=True)["meta"]
        assert meta["fold"] == "Handwriting" and len(meta["ctx_mu"]) == len(g.ET_COLS)
        mu, _sd = m33.table_standardizer(table, sorted({r["task_id"] for r in sub if r["dataset"] != "Handwriting"}), g.ET_COLS)
        assert np.allclose(meta["ctx_mu"], mu)                                        # 只用子采样后的训练切片
