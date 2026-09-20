"""WP17-ACTS S1″（results/acts/s1pp/PROTOCOL.md）——先于实现。覆盖：profile extends、剪枝谓词 gte、G-A0 regret 门、
钉底尺度 / 前一半噪声 / 净占比、准入 v2、配对权重与产量、shared-30 前缀复用、整集官方划分数组、按格分发执行体、
第 2 步计划与决策 W、第 3 步计划、第 4 步决策 H、候选池注册表回退。零 GPU、不读真实数据。"""
from __future__ import annotations

import importlib

import numpy as np
import pytest


def _ac():
    return importlib.import_module("scripts.acts_common")


def _g83():
    return importlib.import_module("scripts.83_acts_corpus")


@pytest.fixture(scope="module")
def cfg():
    return _ac().load_acts_config(profile="s1pp")


@pytest.fixture(scope="module")
def cfg_p():
    return _ac().load_acts_config(profile="s1p")


# ---------------------------------------------------------------- profile / 剪枝（§1.3）
class TestProfileAndPrune:
    def test_extends_s1p_then_overrides(self, cfg, cfg_p):
        assert cfg["profile"] == "s1pp" and cfg["eval"]["mode"] == "cv5" and cfg["slicing"]["n_min"] == 80      # 来自 s1p
        assert cfg["candidates"]["n_shared"] == 30 and cfg["labels"]["floor_mode"] == "diagnostic"            # 本块覆盖
        assert [r["name"] for r in cfg["prune"]["rules"]] == ["tau100", "infonce_tau10", "tel_tau_ge1", "triplet_tau10"]
        assert cfg_p["candidates"]["n_shared"] == 20 and len(cfg_p["prune"]["rules"]) == 2 and "extends" not in cfg_p
        assert cfg["admission"]["rule"] == "v2" and cfg["admission"]["task_rows"] == [1, 7, 3, 5]

    def test_prune_gte(self, cfg, cfg_p):
        from autocls.space.cls_space import CLS
        ac = _ac()
        base = dict(resize=0.0, rescale=0.0, jitter=0.0, point_mask=0.0, freq_mask=0.0, crop=0.5, order=0, emb_jitter=0.0, emb_mask=0.0,
                    instance=True, temporal=False, cross_scale=False, kernel=None, pool=None, adj_neighbor=None)
        mk = lambda **kw: CLS(**{**base, **kw})
        assert ac.is_pruned(mk(loss_type="triplet", sim_func="euclidean", norm="l2", temperature=1.0), cfg)
        assert ac.is_pruned(mk(loss_type="triplet", sim_func="euclidean", norm="l2", temperature=10.0), cfg)
        assert not ac.is_pruned(mk(loss_type="triplet", sim_func="euclidean", norm="l2", temperature=0.1), cfg)
        assert ac.is_pruned(mk(loss_type="triplet", sim_func="dot", norm="none", temperature=10.0), cfg)
        assert not ac.is_pruned(mk(loss_type="info_nce", sim_func="dot", norm="none", temperature=1.0), cfg)
        assert ac.is_pruned(mk(loss_type="info_nce", sim_func="dot", norm="none", temperature=100.0), cfg)
        assert not ac.is_pruned(mk(loss_type="triplet", sim_func="euclidean", norm="l2", temperature=1.0), cfg_p)   # s1p 规则不变

    def test_pruning_regret_gate(self, cfg):
        ac = _ac()
        c_ok = {"loss_type": "info_nce", "sim_func": "dot", "norm": "none", "temperature": 1.0}
        c_bad = {"loss_type": "triplet", "sim_func": "dot", "norm": "none", "temperature": 10.0}
        fields = dict(resize=0.0, rescale=0.0, jitter=0.0, point_mask=0.0, freq_mask=0.0, crop=0.5, order=0, emb_jitter=0.0, emb_mask=0.0,
                      instance=True, temporal=False, cross_scale=False, kernel=None, pool=None, adj_neighbor=None)
        rows = [{"stage": "s1_catalog", "seed": 0, "dataset": "A", "cls_key": "a", "cls": {**fields, **c_bad}, "acc": 0.90, "error": None},
                {"stage": "s1_catalog", "seed": 0, "dataset": "A", "cls_key": "b", "cls": {**fields, **c_ok}, "acc": 0.88, "error": None},
                {"stage": "s1_catalog", "seed": 0, "dataset": "B", "cls_key": "b", "cls": {**fields, **c_ok}, "acc": 0.70, "error": None},
                {"stage": "s1_catalog", "seed": 0, "dataset": "B", "cls_key": "a", "cls": {**fields, **c_bad}, "acc": 0.60, "error": None},
                {"stage": "s1_catalog", "seed": 1, "dataset": "B", "cls_key": "a", "cls": {**fields, **c_bad}, "acc": 0.99, "error": None}]  # seed 1 不计
        r = ac.pruning_regret(rows, cfg)
        assert r["n"] == 2 and r["n_lost"] == 1 and r["lost"][0][0] == "A" and abs(r["mean_gap"] - 0.01) < 1e-9 and abs(r["max_gap"] - 0.02) < 1e-9
        assert r["ok"] is True and ac.pruning_regret(rows, {**cfg, "prune": {**cfg["prune"], "regret_gate": {"mean_max": 0.005}}})["ok"] is False


# ---------------------------------------------------------------- 尺度（§1.2）
class TestScale:
    def test_clip_z_median(self):
        ac = _ac()
        v = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        z, mu, sd, cut = ac.clip_z(v, 0.5)
        assert cut == pytest.approx(0.35) and z.shape == (6,) and np.allclose(z[:3], z[0]) and z[5] > z[4] > z[3] > z[0]
        assert abs(z.mean()) < 1e-9 and abs(z.std() - 1) < 1e-9

    def test_top_half_sigma(self):
        ac = _ac()
        a0 = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1]); a1 = a0 + np.array([0.02, -0.02, 0.02, -0.02, 0.5, -0.5])   # 后一半噪声大
        s, sm, n = ac.top_half_sigma(a0, a1, top_frac=0.5)
        assert n == 3 and s == pytest.approx(0.02 / np.sqrt(2)) and sm == pytest.approx(s / np.sqrt(2))

    def test_net_interaction_share_pure_cases(self):
        ac = _ac()
        rng = np.random.default_rng(0)
        cfg_eff = rng.normal(size=8)
        Z_main = np.tile(cfg_eff, (6, 1))                                    # 纯主效应
        r = ac.net_interaction_share(Z_main, Z_main, Z_main)
        assert r["net_share"] == pytest.approx(0.0, abs=1e-9) and r["gross_share"] == pytest.approx(0.0, abs=1e-9)
        Z_int = rng.normal(size=(6, 8)); Z_int -= Z_int.mean(0, keepdims=True); Z_int -= Z_int.mean(1, keepdims=True)   # 纯交互、无噪声
        r = ac.net_interaction_share(Z_int, Z_int, Z_int)
        assert r["gross_share"] == pytest.approx(1.0) and r["net_share"] == pytest.approx(1.0) and r["real_share"] == pytest.approx(1.0)
        Z0 = Z_int + rng.normal(scale=0.5, size=Z_int.shape); Z1 = Z_int + rng.normal(scale=0.5, size=Z_int.shape)
        r = ac.net_interaction_share((Z0 + Z1) / 2, Z0, Z1)
        assert 0 < r["ms_noise"] < r["ms_int"] and r["net_share"] < r["gross_share"] and r["real_share"] < 1.0


# ---------------------------------------------------------------- 准入 v2（§1.5）
class TestAdmissionV2:
    def _source(self, rng, n_tasks=4, n_shared=30, structured=True, noise=0.01, scale=0.3):
        base = np.sort(rng.uniform(0.3, 0.9, n_shared))[::-1]
        tasks = {}
        for t in range(n_tasks):
            m = base + (rng.normal(scale=scale, size=n_shared) if structured else 0.0)   # 任务特异 0.3 → 净占比 ≈ 0.39；无任务特异 → ≈ 0
            a0 = m + rng.normal(scale=noise, size=n_shared); a1 = m + rng.normal(scale=noise, size=n_shared)
            tasks[f"S_a{t:02d}"] = (a0, a1)
        return tasks

    def test_metrics_and_decision(self, cfg):
        ac = _ac()
        rng = np.random.default_rng(1)
        good = ac.admission_v2_metrics(self._source(rng, structured=True), clip_quantile=0.5, noise_top_frac=0.5)
        flat = ac.admission_v2_metrics(self._source(rng, structured=False, noise=0.05), clip_quantile=0.5, noise_top_frac=0.5)
        assert set(good) >= {"rho_top", "net_share", "iqr_sigma", "n_tasks"}
        assert good["net_share"] > flat["net_share"] and good["iqr_sigma"] > flat["iqr_sigma"] and good["rho_top"] > flat["rho_top"]
        d = ac.admission_v2_decision({"G": good, "F": flat, "X": {"rho_top": 0.9, "net_share": 0.9, "iqr_sigma": 9.0, "n_tasks": 2}}, cfg)
        assert d["admitted"] == ["G"] and "F" in d["rejected"] and "X" in d["rejected"] and d["below_floor"] is True
        assert d["reasons"]["X"] == ["min_tasks"]

    def test_task_rows_1735(self, cfg):
        g = _g83()
        meta = {f"Q_a{r:02d}": {"dataset": "Q"} for r in range(8)}
        assert g.admission_tasks({**cfg, "admission": {**cfg["admission"], "candidates": ["Q"]}}, meta) == {"Q": ["Q_a01", "Q_a07", "Q_a03", "Q_a05"]}


# ---------------------------------------------------------------- 配对权重（§1.1）
class TestPairs:
    def test_weights_dead_zone_cap_boost(self, cfg):
        ac = _ac()
        vals = np.array([0.90, 0.89, 0.80, 0.50]); n_seeds = np.array([2, 2, 1, 1])
        W = ac.pair_weights(vals, n_seeds, sigma_task=0.02, **cfg["labels"]["pairs"])
        assert W.shape == (4, 4) and np.allclose(W, W.T) and np.all(np.diag(W) == 0)
        assert W[0, 1] == 0                                              # gap 0.01 < 2·sd_gap(0.02)
        assert W[0, 3] == 2.0 and W[2, 3] == 1.0                         # 封顶 1，含前四分位（0.90）的对 ×2
        assert 0 < W[1, 2] < 2.0
        assert ac.pair_yield(W) == 5                                     # 只有 (0, 1) 落在死区

    def test_shared30_prefix_reuses_s1p_unpruned(self, cfg, cfg_p):
        ac = _ac()
        sp, spp = ac.acts_space(cfg_p), ac.acts_space(cfg)
        old = ac.shared_catalog(sp, cfg_p)
        new = ac.shared_catalog(spp, cfg)
        keep = [k for c, k in old if not ac.is_pruned(c, cfg)]
        assert len(old) == 20 and len(new) == 30 and len({k for _c, k in new}) == 30
        assert [k for _c, k in new[: len(keep)]] == keep and 15 <= len(keep) <= 19
        assert all(not ac.is_pruned(c, cfg) for c, _k in new)


# ---------------------------------------------------------------- 整集数组 / 分发执行体 / 第 2 步（§1.7）
class TestWhole:
    def test_whole_arrays_official(self):
        ac = _ac()
        rng = np.random.default_rng(0)
        X = rng.normal(loc=3.0, scale=2.0, size=(50, 12, 3)).astype(np.float32); y = np.repeat(np.arange(5), 10)
        a = ac.whole_arrays_official(X, y, n_train=30)
        assert a["Xtr"].shape == (30, 12, 3) and a["Xte_train"].shape == (20, 12, 3) and a["yte"].shape == (20,)
        assert np.allclose(a["Xtr"].reshape(-1, 3).mean(0), 0, atol=1e-5) and np.allclose(a["Xtr"].reshape(-1, 3).std(0), 1, atol=1e-4)
        assert a["train_idx"].tolist() == list(range(30)) and a["eval_idx"].tolist() == list(range(30, 50)) and 0 < a["majority"] <= 1
        assert "Xte_split" in a and a["Xte_split"].shape == a["Xte_train"].shape

    def test_dispatch_runner(self, monkeypatch):
        g = _g83()
        monkeypatch.setattr(g._g39, "run_trajectory_cell", lambda cell, arrays, device: {"which": "official"})
        monkeypatch.setattr(g, "run_trajectory_cell_cv", lambda cell, arrays, device: {"which": "cv5"})
        assert g.dispatch_runner({"eval": "official"}, {}, "cpu") == {"which": "official"}
        assert g.dispatch_runner({"eval": "cv5"}, {}, "cpu") == {"which": "cv5"}
        with pytest.raises(ValueError):
            g.dispatch_runner({"eval": "holdout7"}, {}, "cpu")

    def test_build_plan_w(self, cfg):
        g = _g83()
        sizes = {"Cricket": {"n_train": 108, "n_test": 72, "L": 1197, "C": 6}, "UWaveGestureLibrary": {"n_train": 120, "n_test": 320, "L": 315, "C": 3}}
        shared = [({"k": i}, f"key{i:02d}") for i in range(20)]
        plan = g.build_plan_w(cfg, sizes, shared_keys=[k for _c, k in shared], shared_cls={k: _fake_cls(i) for i, (_c, k) in enumerate(shared)})
        cells = plan["cells"]
        assert plan["n_cells"] == 160 and len({c["key"] for c in cells}) == 160 and sorted({c["eval"] for c in cells}) == ["cv5", "official"]
        off = [c for c in cells if c["task_id"] == "Cricket_whole_official"]
        cv = [c for c in cells if c["task_id"] == "Cricket_whole_cv5"]
        assert len(off) == 40 and off[0]["n_train"] == 108 and cv[0]["n_train"] == 180 and off[0]["iters"] == 600 and cv[0]["T"] == 1197
        assert plan["probe_tasks"] == ["Cricket_whole_official", "Cricket_whole_cv5", "UWaveGestureLibrary_whole_official", "UWaveGestureLibrary_whole_cv5"]
        assert plan["n_probe"] == 12

    def test_step2_decision(self, cfg):
        ac = _ac()
        assert ac.step2_decision(0.3, 0.8, cfg) == {"code": "W-PROTOCOL", "eval_mode": "holdout", "nl_effect": False}
        assert ac.step2_decision(0.7, 0.3, cfg) == {"code": "W-NL", "eval_mode": "cv5", "nl_effect": True}
        assert ac.step2_decision(0.7, 0.8, cfg) == {"code": "W-CONSISTENT", "eval_mode": "cv5", "nl_effect": False}
        assert ac.step2_decision(0.3, 0.3, cfg)["eval_mode"] == "holdout" and ac.step2_decision(0.3, 0.3, cfg)["nl_effect"] is True


def _fake_cls(i: int):
    from autocls.space.cls_space import CLS
    return CLS(resize=0.0, rescale=0.0, jitter=0.0, point_mask=0.0, freq_mask=0.0, crop=0.5, order=0, emb_jitter=0.0, emb_mask=0.0,
               norm="none", instance=True, temporal=False, cross_scale=False, kernel=None, pool=None, adj_neighbor=None,
               loss_type="info_nce", sim_func="dot", temperature=[0.01, 0.1, 1.0][i % 3])


# ---------------------------------------------------------------- 第 3 步计划 / 第 4 步决策（§1.8）
class TestPlanCAndHeadroom:
    def test_build_plan_c_counts(self, cfg):
        g = _g83()
        srcs = ["PhonemeSpectra", "Cricket", "UWaveGestureLibrary", "Handwriting", "PTB-XL"]
        meta = {}
        for s in srcs:
            for r in range(8):
                meta[f"{s}_a{r:02d}"] = {"dataset": s, "n_train": 200, "T": 100, "C": 3, "k": 5, "majority": 0.3, "iters": 200, "tag": f"c0w0n{r % 4}", "time_window": [0, 100]}
        c2 = {**cfg, "admission": {**cfg["admission"], "candidates": srcs}}
        shared = [(_fake_cls(i), f"key{i:02d}") for i in range(30)]
        plan = g.build_plan_c(c2, meta, shared=shared, existing_keys=set())
        cells = plan["cells"]
        # 试点 3 集 × 8 任务 + 非试点 2 源 × 4 任务 = 32 任务 × 30 × 2 种子
        assert plan["n_cells"] == 32 * 60 and len({c["key"] for c in cells}) == plan["n_cells"] and all(c["arm"] == "shared" for c in cells)
        assert sorted({c["task_id"] for c in cells if c["task_id"].startswith("PTB-XL")}) == ["PTB-XL_a01", "PTB-XL_a03", "PTB-XL_a05", "PTB-XL_a07"]
        assert len({c["task_id"] for c in cells if c["task_id"].startswith("Cricket")}) == 8
        done = {c["key"] for c in cells[:100]}
        plan2 = g.build_plan_c(c2, meta, shared=shared, existing_keys=done)
        assert plan2["n_cells"] == plan["n_cells"] and plan2["n_new"] == plan["n_cells"] - 100 and plan2["n_existing"] == 100

    def test_headroom_decision(self, cfg):
        ac = _ac()
        assert ac.headroom_decision(0.035, cfg) == "HEADROOM-GO"
        assert ac.headroom_decision(0.015, cfg) == "HEADROOM-B"
        assert ac.headroom_decision(0.025, cfg) == "HEADROOM-MID"
        assert ac.headroom_decision(float("nan"), cfg) == "HEADROOM-NA"

    def test_honest_headroom_matrix(self):
        ac = _ac()
        rng = np.random.default_rng(0)
        base = np.linspace(0.9, 0.5, 10)
        S0 = np.tile(base, (6, 1)) + rng.normal(scale=0.001, size=(6, 10)); S1 = np.tile(base, (6, 1)) + rng.normal(scale=0.001, size=(6, 10))
        groups = np.array(["A", "A", "B", "B", "C", "C"])
        r = ac.honest_headroom(S0, S1, groups)
        assert abs(r["regret1_mean"]) < 0.01 and abs(r["regret3_mean"]) < 0.01 and r["n_tasks"] == 6        # 无任务特异 → 余量 ≈ 0
        S0b = S0.copy(); S1b = S1.copy(); S0b[0, 9] = S1b[0, 9] = 0.99                                       # 任务 0 的 oracle 是全局最差的配置
        r2 = ac.honest_headroom(S0b, S1b, groups)
        assert r2["regret1_mean"] > 0.05 / 6 and r2["per_task"][0] == pytest.approx(0.99 - S1b[0, 0], abs=0.01)


# ---------------------------------------------------------------- 分片：--resume 时只按未完成的格均分（磁盘写满后重启不再把已完成的任务算进负载）
def test_part_tasks_balance_pending_only(cfg):
    g = _g83()
    cells = [{"task_id": f"T{i}", "key": f"k{i}_{s}", "iters": 600, "T": 100, "C": 3, "n_train": 200} for i in range(6) for s in (0, 1)]
    done = {c["key"] for c in cells if c["task_id"] in ("T0", "T1", "T2", "T3")}          # 前 4 个任务已跑完
    parts = g.part_tasks(cells, done, 2, cfg)
    assert sorted(parts[0] + parts[1]) == ["T4", "T5"] and len(parts[0]) == 1 and len(parts[1]) == 1
    assert sorted(sum(g.part_tasks(cells, set(), 2, cfg), [])) == [f"T{i}" for i in range(6)]   # 无完成记录 → 与原分法同（全部任务）
    assert sorted(sum(g.part_tasks(cells, {c["key"] for c in cells}, 2, cfg), [])) == []        # 全部完成 → 空


# ---------------------------------------------------------------- 注册表回退
def test_load_dataset_full_candidate_pool_fallback(monkeypatch):
    import autocls.data.source_pool as sp
    ac = _ac()
    seen = []
    monkeypatch.setattr(sp, "load_source_dataset", lambda name, reg, crop_len="auto": (seen.append((name, reg.get(name).source, crop_len)) or (np.zeros((4, 8, 2), np.float32), np.arange(4))))
    X, y = ac.load_dataset_full("PTB-XL")
    assert seen == [("PTB-XL", "npz", None)] and X.shape == (4, 8, 2)
    ac.load_dataset_full("Opportunity")
    assert seen[-1] == ("Opportunity", "npz", None)
    with pytest.raises(KeyError):
        ac.load_dataset_full("NoSuchDataset")
