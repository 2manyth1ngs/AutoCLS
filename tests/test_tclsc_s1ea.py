"""WP16-TCLSC 阶段 1 响应画像 E（A 臂）——先于实现（铁律 5）。协议 results/tclsc_s1ea/PROTOCOL.md。

覆盖两层：
- `autocls/data/mech_features.py`（自 wp11-mech 移植 + WP14 v2 口径）：确定性变换 op、信息量 info、保持率两种形式（within / cross）、
  无标签失真、结构特征、`response_profile` 的 25 维列表与 task_type 分发（铁律 7）。
- `scripts/63_tclsc_s1ea.py`：臂表、nan 感知的折内标准化（只用训练集）、ctx 表、字段偏好 / 机制对齐、门与读数码、参照臂排名合并、
  端到端小样本训练 → meta / 排名。
零 GPU、不读真实缓存。
"""
from __future__ import annotations

import importlib
import json

import numpy as np
import pytest
import torch

from autocls.space.cls_space import CLSSpace
from autocls.task_types import TASK_CLASSIFICATION
from scripts import pq_common as pq

m33 = importlib.import_module("scripts.33_pq_comparator_loo")
SPACE = CLSSpace.from_yaml()


def _mf():
    return importlib.import_module("autocls.data.mech_features")


def _g63():
    return importlib.import_module("scripts.63_tclsc_s1ea")


def _g60():
    return importlib.import_module("scripts.60_tclsc_s1ucr")


def _blobs(n_per=30, T=64, C=1, k=3, sep=6.0, seed=0):
    """k 个类各 n_per 条，类中心相隔 sep 的高斯团（原始空间 1-NN 可分）。"""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(k):
        center = rng.standard_normal((T, C)) * sep
        X.append(center[None] + rng.standard_normal((n_per, T, C)))
        y += [c] * n_per
    return np.concatenate(X).astype(np.float32), np.array(y)


# ================================================================ mech_features
class TestOps:
    def test_module_imports_without_torch(self):
        mf = _mf()
        assert "torch" not in vars(mf)

    def test_deterministic_ops(self):
        mf = _mf()
        x = torch.arange(1, 9, dtype=torch.float32).view(1, 8, 1) * 0.5      # 严格线性
        op = mf.deterministic_op
        n = op("inst_norm")(x, 0.0, None)
        assert torch.allclose(n.mean(dim=1), torch.zeros(1, 1), atol=1e-5) and torch.allclose(n.std(dim=1, unbiased=False), torch.ones(1, 1), atol=1e-3)
        assert torch.allclose(op("detrend")(x, 0.0, None), torch.zeros_like(x), atol=1e-5)
        d = op("diff1")(x, 0.0, None)
        assert d[0, 0, 0] == 0 and torch.allclose(d[0, 1:, 0], torch.full((7,), 0.5))
        assert torch.equal(op("polarity")(x, 0.0, None), -x)
        assert torch.equal(op("time_reverse")(x, 0.0, None), torch.flip(x, dims=[1]))
        hp = op("highpass05")(x, 0.0, None)
        assert abs(float(hp.mean())) < 1e-5                                    # DC 必去
        const = torch.ones(2, 16, 1)
        assert torch.allclose(op("lowpass25")(const, 0.0, None), const, atol=1e-5)
        with pytest.raises(ValueError):
            op("smooth7")

    def test_op_contract_accepts_tensor_or_tuple(self):
        mf = _mf()
        X, y = _blobs()
        r1 = mf.label_retention(X, y, lambda x, p, generator=None: x, 0.5)
        r2 = mf.label_retention(X, y, lambda x, p, generator=None: (x, None), 0.5)
        assert r1["retention"] == pytest.approx(1.0) and r2["retention"] == pytest.approx(1.0)


class TestInfo:
    def test_info_loo_separable_is_one(self):
        mf = _mf()
        X, y = _blobs()
        assert mf.info_loo(X.reshape(len(y), -1), y) == pytest.approx(1.0)
        assert mf.info_loo(np.random.default_rng(0).standard_normal((60, 5)), np.arange(60) % 2) <= 1.0

    def test_retention_info_forms(self):
        mf = _mf()
        X, y = _blobs()
        ident = lambda x, p, generator=None: x
        w = mf.retention_info(X, y, ident, 0.0, form="within")
        c = mf.retention_info(X, y, ident, 0.0, form="cross", seeds=(0, 1))
        assert w["delta"] == pytest.approx(0.0) and c["delta"] == pytest.approx(0.0)
        assert w["info_raw"] == pytest.approx(1.0) and set(w) >= {"info_raw", "info_after", "delta"}
        # 极端破坏：把全部样本换成同一个常数 → within 信息量落到 ≤ 0，delta ≤ −1 + ε
        kill = lambda x, p, generator=None: torch.zeros_like(x)
        assert mf.retention_info(X, y, kill, 0.0, form="within")["delta"] <= -0.9
        with pytest.raises(ValueError):
            mf.retention_info(X, y, ident, 0.0, form="both")

    def test_random_ops_use_seeds_deterministically(self):
        mf = _mf()
        from autocls.contrastive.augmentations import jitter
        X, y = _blobs()
        a = mf.retention_info(X, y, jitter, 0.5, form="cross", seeds=(0, 1, 2))
        b = mf.retention_info(X, y, jitter, 0.5, form="cross", seeds=(0, 1, 2))
        assert a["delta"] == b["delta"] and -1.0 <= a["delta"] <= 1.0

    def test_distortion_cos(self):
        mf = _mf()
        X, _y = _blobs()
        assert mf.distortion_cos(X, lambda x, p, generator=None: x, 0.0) == pytest.approx(1.0)
        assert mf.distortion_cos(X, mf.deterministic_op("polarity"), 0.0) == pytest.approx(-1.0)
        assert -1.0 <= mf.distortion_cos(X, mf.deterministic_op("diff1"), 0.0) <= 1.0


class TestStructural:
    def test_linear_fit_and_trend(self):
        mf = _mf()
        t = np.arange(32, dtype=np.float32)
        X = np.stack([np.stack([2 * t + 1, np.ones(32)], axis=1)])              # (1, 32, 2)：一条严格线性、一条常数
        slope, mean, r2 = mf.linear_fit(X)
        assert slope.shape == (1, 2) and slope[0, 0] == pytest.approx(2.0) and slope[0, 1] == pytest.approx(0.0)
        assert r2[0, 0] == pytest.approx(1.0) and r2[0, 1] == pytest.approx(0.0)
        assert mf.trend_strength(X) == pytest.approx(0.5)

    def test_level_scale_and_trend_info_range(self):
        mf = _mf()
        X, y = _blobs()
        for v in (mf.level_scale_info(X, y), mf.trend_info(X, y)):
            assert -1.0 <= v <= 1.0
        h = mf.hardness_1nn(X, y)
        assert h["hardness"] == pytest.approx(0.0) and h["nn1_base_acc"] == pytest.approx(1.0)


class TestResponseProfile:
    def test_cols_and_dispatch(self):
        mf = _mf()
        cols = mf.RESPONSE_PROFILE_COLS
        assert len(cols) == 25 and len(set(cols)) == 25
        assert sum(c.startswith("ret_") for c in cols) == 11 and sum(c.startswith("sim_") for c in cols) == 10
        assert {"hardness_1nn", "level_scale_info", "trend_info", "trend_strength"} <= set(cols)
        assert "sim_polarity" not in cols and "ret_polarity" in cols and "ret_lowpass25" in cols
        assert mf.RESPONSE_OPS["inst_norm"]["form"] == "within" and mf.RESPONSE_OPS["jitter"]["form"] == "cross"
        assert mf.RESPONSE_OPS["jitter"]["p"] == 0.5 and mf.RESPONSE_OPS["highpass05"]["p"] == 0.0
        X, y = _blobs(n_per=12, T=32)
        with pytest.raises(NotImplementedError):
            mf.response_profile(X, y, task_type="forecasting")
        with pytest.raises(ValueError):
            mf.response_profile(X, y, task_type="regression")

    def test_profile_values(self):
        mf = _mf()
        X, y = _blobs(n_per=12, T=32)
        prof = mf.response_profile(X, y, task_type=TASK_CLASSIFICATION, seeds=(0, 1))
        assert set(mf.RESPONSE_PROFILE_COLS) <= set(prof)
        vals = np.array([prof[c] for c in mf.RESPONSE_PROFILE_COLS], dtype=float)
        assert np.all(np.isfinite(vals))
        assert all(-1.0 <= prof[c] <= 1.0 for c in mf.RESPONSE_PROFILE_COLS if c.startswith(("ret_", "sim_")))
        assert prof["n"] == 36 and prof["K"] == 3 and prof["info_raw"] == pytest.approx(1.0)
        assert "sim_polarity" not in prof


# ================================================================ scripts/63
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


def _ea_table(datasets, seed=1, nan_at=None):
    g = _g63()
    rng = np.random.default_rng(seed)
    tab = {d: {"dataset": d, "task_id": f"{d}_00", **{c: float(rng.standard_normal()) for c in g.EA_COLS}} for d in datasets}
    if nan_at:
        d, c = nan_at
        tab[d][c] = None
    return tab


def _phi(datasets, seed=2):
    rng = np.random.default_rng(seed)
    return {d: {"task_id": f"{d}_00", "dataset": d, **{c: float(rng.standard_normal()) for c in m33.PHI_COLS}} for d in datasets}


class TestArms:
    def test_arm_table(self):
        g = _g63()
        assert g.EA_COLS == tuple(_mf().RESPONSE_PROFILE_COLS) and len(g.EA_COLS) == 25
        a = g.ARMS
        assert {"u_ea_diff_hard", "u_ea_concat_hard", "u_eaphi_diff_hard"} <= set(a)   # R 检验又加了 u_fw4_ / u_fw_ 臂
        assert a["u_ea_diff_hard"].ctx_kind == "ea_budget" and a["u_ea_diff_hard"].head == "diff"
        assert a["u_ea_concat_hard"].ctx_kind == "ea_budget" and a["u_ea_concat_hard"].head == "concat"
        assert a["u_eaphi_diff_hard"].ctx_kind == "eaphi_budget" and a["u_eaphi_diff_hard"].head == "diff"
        assert all(s.split == "S1" and s.seeds == (0, 1, 2) and s.label_mode == "hard" for s in a.values())
        assert g.ctx_cols("ea_budget") == g.EA_COLS and g.ctx_cols("eaphi_budget") == g.EA_COLS + tuple(m33.PHI_COLS)
        assert g.REF_ARMS == ("u_blind_diff_hard", "u_phi_diff_hard")


class TestCtx:
    def test_standardizer_nan_aware_train_only(self):
        g = _g63()
        ds = ["DS00", "DS01", "DS02", "DS03"]
        tab = _ea_table(ds, nan_at=("DS01", g.EA_COLS[0]))
        mu, sd = g.table_standardizer(tab, ["DS00", "DS01", "DS02"], g.EA_COLS)
        assert mu.shape == (25,) and sd.shape == (25,) and np.all(sd >= m33.PHI_SD_FLOOR)
        c0 = g.EA_COLS[0]
        assert mu[0] == pytest.approx(np.mean([tab["DS00"][c0], tab["DS02"][c0]]))         # nan 不进均值、DS03 不进
        assert mu[1] == pytest.approx(np.mean([tab[d][g.EA_COLS[1]] for d in ("DS00", "DS01", "DS02")]))

    def test_table_context_shapes_and_nan_fill(self):
        g = _g63()
        ds = ["DS00", "DS01", "DS02", "DS03"]
        tab = _ea_table(ds, nan_at=("DS03", g.EA_COLS[2]))
        ctx, meta = g.table_context(tab, g.EA_COLS, ["DS00", "DS01", "DS02"], ["DS03"], ctx_kind="ea_budget")
        tid = pq.engine_task_id("DS03_00", 120)
        assert ctx[tid].shape == (25 + 3,) and ctx[tid][2] == 0.0                              # nan → 训练均值 → 标准化后 0
        assert ctx[tid][25:].tolist() == pq.ctx_vector(120, pq.BUDGETS).tolist()
        assert meta["ctx_kind"] == "ea_budget" and len(meta["ctx_mu"]) == 25 and meta["ctx_cols"] == list(g.EA_COLS)
        assert meta["n_nan"] == {"DS03": 1}
        # 与 φ 叠加：37 + 3
        both = g.merge_tables(tab, _phi(ds))
        ctx2, meta2 = g.table_context(both, g.ctx_cols("eaphi_budget"), ["DS00", "DS01", "DS02"], ["DS03"], ctx_kind="eaphi_budget")
        assert ctx2[tid].shape == (37 + 3,) and len(meta2["ctx_mu"]) == 37

    def test_ctx_rows_cross_zero(self):
        g = _g63()
        ds = ["DS00", "DS01", "DS02", "DS03"]
        tab = _ea_table(ds)
        mu, sd = g.table_standardizer(tab, ds[:3], g.EA_COLS)
        rows = g.table_ctx_rows(tab, g.EA_COLS, ["DS03_00"], mu, sd)
        assert set(rows) == {("DS03_00", b) for b in pq.BUDGETS} and rows[("DS03_00", 60)].shape == (28,)
        z = g.table_ctx_rows(tab, g.EA_COLS, ["DS03_00"], mu, sd, zero=True)
        assert np.all(z[("DS03_00", 60)][:25] == 0) and z[("DS03_00", 60)][25] == 1.0
        cm = {"DS03_00": "DS01_00"}
        x = g.table_ctx_rows(tab, g.EA_COLS, ["DS03_00"], mu, sd, cross_map=cm)
        own = g.table_ctx_rows(tab, g.EA_COLS, ["DS01_00"], mu, sd)
        assert np.allclose(x[("DS03_00", 290)][:25], own[("DS01_00", 290)][:25])


class TestAlignment:
    def test_field_preference_and_alignment(self):
        g = _g63()
        rows = _ucr_rows(n_datasets=6)
        cat_z = _g60()._g56.zscore_within(rows)
        u = g.field_preference(cat_z, "jitter", budget=290)
        assert set(u) <= {f"DS{i:02d}" for i in range(6)}
        for v in u.values():
            assert np.isfinite(v)
        # 人造完全对齐：ret_jitter = u_f → ρ = 1
        tab = {d: {c: 0.0 for c in g.EA_COLS} for d in u}
        for d, v in u.items():
            tab[d]["ret_jitter"] = v
        al = g.field_alignment(tab, cat_z, fields=("jitter",), budget=290)
        assert al["jitter"]["n"] == len(u) and al["jitter"]["rho"] == pytest.approx(1.0)
        assert g.FIELD_OPS == {"jitter": "ret_jitter", "rescale": "ret_rescale", "point_mask": "ret_point_mask", "freq_mask": "ret_freq_mask"}
        assert (g.FIELD_HI, g.FIELD_LO, g.FIELD_MIN_SIDE) == (0.5, 0.2, 2)

    def test_field_preference_needs_both_sides(self):
        g = _g63()
        rows = [r for r in _ucr_rows(n_datasets=1) if float(r["cls"]["jitter"]) <= 0.2]
        cat_z = _g60()._g56.zscore_within(rows)
        assert g.field_preference(cat_z, "jitter", budget=290) == {}


class TestGatesEa:
    def _inp(self, d, p, blind, ea, p2, phi, eaphi, p3, dc=0.0):
        deltas = {"u_ea_diff_hard": {"acc_top10": {"mean_delta": d, "p": p}},
                  "u_ea_concat_hard": {"acc_top10": {"mean_delta": dc, "p": 0.01 if dc else 0.9}}}
        return deltas, {"mean_a": blind, "mean_b": ea, "p": p2}, {"mean_a": phi, "mean_b": eaphi, "p": p3}

    def test_codes(self):
        g = _g63()
        assert g.verdict_ea(g.gates_ea(*self._inp(0.05, 0.01, 0.022, 0.018, 0.01, 0.021, 0.017, 0.01))) == "EA-ALIVE"
        assert g.verdict_ea(g.gates_ea(*self._inp(0.05, 0.01, 0.022, 0.018, 0.01, 0.021, 0.020, 0.5))) == "EA-PHI-EQUIV"
        assert g.verdict_ea(g.gates_ea(*self._inp(0.05, 0.01, 0.022, 0.022, 0.5, 0.021, 0.017, 0.01))) == "EA-USED-NO-GAIN"
        assert g.verdict_ea(g.gates_ea(*self._inp(0.01, 0.01, 0.022, 0.018, 0.01, 0.021, 0.017, 0.01))) == "EA-DEAD"
        assert g.verdict_ea(g.gates_ea(*self._inp(0.05, 0.2, 0.022, 0.018, 0.01, 0.021, 0.017, 0.01))) == "EA-DEAD"
        gt = g.gates_ea(*self._inp(0.05, 0.01, 0.022, 0.018, 0.01, 0.021, 0.017, 0.01, dc=0.05))
        assert gt["GA5"]["pass"] is False and gt["GA1"]["pass"] is True


class TestMergeRef:
    def test_merge_ref_rankings(self):
        g = _g63()
        own = {"fold0": {"normal": {"60": {"DS00_00": {"u_ea_diff_hard": ["a", "b"]}}}, "cross": {}, "zero": {}, "test": ["DS00"]}}
        ref = {"fold0": {"normal": {"60": {"DS00_00": {"u_blind_diff_hard": ["b", "a"], "u_phi_diff_hard": ["a", "b"], "u_fp_diff_hard": ["b", "a"]}}},
                         "cross": {}, "zero": {}, "test": ["DS00"], "probes": ["a"]}}
        merged = g.merge_ref_rankings(own, ref, arms=g.REF_ARMS)
        per = merged["fold0"]["normal"]["60"]["DS00_00"]
        assert set(per) == {"u_ea_diff_hard", "u_blind_diff_hard", "u_phi_diff_hard"}
        with pytest.raises(ValueError):
            g.merge_ref_rankings(own, {"fold0": dict(ref["fold0"], test=["DS01"])}, arms=g.REF_ARMS)


class TestEndToEnd:
    def test_train_and_rank_small(self, tmp_path):
        g = _g63()
        rows = _ucr_rows(n_datasets=4)
        ds = sorted({r["dataset"] for r in rows})
        tab, phi = _ea_table(ds, nan_at=("DS02", g.EA_COLS[5])), _phi(ds)
        folds = [["DS02", "DS03"], ["DS00", "DS01"]]
        outdir = tmp_path / "ea"
        g.train_fold(rows, tab, phi, folds, 0, arm="u_ea_diff_hard", seed=0, outdir=outdir, device="cpu", epochs=2)
        g.train_fold(rows, tab, phi, folds, 0, arm="u_eaphi_diff_hard", seed=0, outdir=outdir, device="cpu", epochs=2)
        meta = torch.load(outdir / "models" / "fold0" / "u_ea_diff_hard_s0.pt", map_location="cpu", weights_only=True)["meta"]
        assert meta["ctx_kind"] == "ea_budget" and meta["ctx_dim"] == 28 and len(meta["ctx_mu"]) == 25 and meta["head"] == "diff"
        assert meta["encoder_mode"] == "binned" and meta["n_nan"] == {"DS02": 1}
        meta2 = torch.load(outdir / "models" / "fold0" / "u_eaphi_diff_hard_s0.pt", map_location="cpu", weights_only=True)["meta"]
        assert meta2["ctx_kind"] == "eaphi_budget" and meta2["ctx_dim"] == 40
        run = json.loads((outdir / "runs" / "fold0" / "u_ea_diff_hard_s0.json").read_text(encoding="utf-8"))
        assert run["delta"] == _g60().DELTA_UCR and run["settings"]["ctx_dim"] == 28
        rk = g.rank_fold(rows, tab, phi, folds, 0, arms=["u_ea_diff_hard", "u_eaphi_diff_hard"], seeds=[0], outdir=outdir)
        keys = sorted({r["cls_key"] for r in rows})
        for name in ("DS02_00", "DS03_00"):
            for b in pq.BUDGETS:
                got = rk["normal"][str(b)][name]
                assert sorted(got["u_ea_diff_hard"]) == keys and sorted(got["u_eaphi_diff_hard"]) == keys
                assert sorted(rk["zero"][str(b)][name]["u_ea_diff_hard"]) == keys
        assert set(rk["cross"]) == {"0", "1", "2"} and rk["test"] == ["DS02", "DS03"]
        for cs in rk["cross"].values():
            assert all(pq.dataset_of(v) != pq.dataset_of(k) for k, v in cs["map"].items())
