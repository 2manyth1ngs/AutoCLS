"""WP17-ACTS S1（results/acts/s1/PROTOCOL.md）——先于实现。覆盖：约束引擎泛化（kernel=0 门控、空间计数）、
SlicedTask.time_window、AutoCTSTierSlicer（三轴四档 OA(16, 4³)、对数样本档、按类分层、前缀稳定）、剪枝谓词、shared / random 目录、锚规则、best-of-3 + 失败地板、
格构造（官方 options / iters 规则 / 锚 seed）、切片数组（训练部分标准化）、G-A 读数量（对半分 / 交互份额）、保真度表、
冠军保留率、读数码、切片 sanity 覆盖表。零 GPU、不读真实数据。"""
from __future__ import annotations

import dataclasses
import importlib
import itertools
import json

import numpy as np
import pytest

from autocls.data.task_slicer import SlicedTask, materialize_task
from autocls.labeling import cls_key
from autocls.space.cls_space import CLS, CLSSpace

GATE_FIELDS = ("temporal", "cross_scale", "kernel", "pool", "adj_neighbor")


def _ac():
    return importlib.import_module("scripts.acts_common")


def _sl():
    return importlib.import_module("autocls.data.acts_slicer")


@pytest.fixture(scope="module")
def cfg():
    return _ac().load_acts_config()


@pytest.fixture(scope="module")
def aspace(cfg):
    return _ac().acts_space(cfg)


@pytest.fixture(scope="module")
def dspace():
    return CLSSpace.from_yaml()


def _canonical_count(space: CLSSpace) -> int:
    """受约束五字段 64 种原始写法经 inactive_overrides 后的规范形个数（独立于 space_size 的再推导）。"""
    forms = set()
    for combo in itertools.product(*[space.choices(n) for n in GATE_FIELDS]):
        values = dict(zip(GATE_FIELDS, combo))
        values.update(space.inactive_overrides(values))
        forms.add(tuple(values[n] for n in GATE_FIELDS))
    return len(forms)


# ---------------------------------------------------------------- 约束引擎（PROTOCOL §1.3）
class TestActsSpace:
    def test_kernel0_canonical_and_default_space_unchanged(self, aspace, dspace, make_cls):
        rng = np.random.default_rng(0)
        saw_k0 = False
        for _ in range(3000):
            c = aspace.sample_random(rng)
            assert aspace.validate(c)
            if c.kernel == 0:
                saw_k0 = True
                assert c.pool is None and c.cross_scale is False
        assert saw_k0
        raw = make_cls(temporal=True, cross_scale=True, kernel=0, pool="avg", adj_neighbor=True)
        assert dspace.validate(raw)                       # 默认空间照旧合法（旧路径逐位不变）
        assert not aspace.validate(raw)                   # acts 空间：非规范形非法
        assert aspace.validate(dataclasses.replace(raw, pool=None, cross_scale=False))
        # 链式触发：temporal=False ∧ kernel=0 → cross_scale=false → 第一条约束 → kernel/pool/adj 全 null
        v = {n: getattr(make_cls(temporal=False, cross_scale=True, kernel=0, pool="max", adj_neighbor=False), n)
             for n in aspace.all_field_names}
        ov = aspace.inactive_overrides(v)
        assert ov == {"pool": None, "cross_scale": False, "kernel": None, "adj_neighbor": None}
        v.update(ov)
        assert CLS(**v) == make_cls()                     # = 旧空间"只有 instance"的规范形

    def test_sampling_stream_identical_to_default(self, aspace, dspace):
        ra, rd = np.random.default_rng(123), np.random.default_rng(123)
        n_same = 0
        for _ in range(300):
            a, d = aspace.sample_random(ra), dspace.sample_random(rd)
            if d.kernel != 0:
                assert a == d
                n_same += 1
        assert n_same > 200

    def test_space_size_enumeration(self, aspace, dspace):
        free = 1
        for n in dspace.field_names:
            if n not in GATE_FIELDS:
                free *= len(dspace.choices(n))
        assert _canonical_count(dspace) == 49 and dspace.space_size() == free * 49
        assert _canonical_count(aspace) == 39 and aspace.space_size() == free * 39


# ---------------------------------------------------------------- 切片（PROTOCOL §1.1 / §1.9：三轴 OA(16, 4³)）
class TestSlicer:
    def test_tier_candidates(self):
        tc = _sl().tier_candidates
        assert [tc(1, 3, i) for i in range(4)] == [[1], [2], [2], [3]]
        assert [tc(1, 11, i) for i in range(4)] == [[1, 2, 3], [4, 5], [6, 7, 8], [9, 10, 11]]
        assert [tc(1, 6, i) for i in range(4)] == [[1, 2], [3], [4], [5, 6]]
        assert [tc(1, 10, i) for i in range(4)] == [[1, 2, 3], [4, 5], [6, 7], [8, 9, 10]]
        parts = [tc(50, 400, i) for i in range(4)]
        assert parts[0][0] == 50 and parts[-1][-1] == 400
        assert sorted(x for p in parts for x in p) == list(range(50, 401))
        assert [tc(50, 50, i) for i in range(4)] == [[50]] * 4

    def test_log_tier_candidates(self):
        ltc = _sl().log_tier_candidates
        parts = [ltc(78, 1024, i) for i in range(4)]          # PS：k = 39 → n_lo 78
        assert parts[0][0] == 78 and parts[-1][-1] == 1024
        assert sorted(x for p in parts for x in p) == list(range(78, 1025))
        # 对数等分：各档上界 ≈ 78 · (1024/78)^{(i+1)/4}
        r = 1024 / 78
        for i in range(3):
            assert parts[i][-1] == max(k for k in range(78, 1025) if k < 78 * r ** ((i + 1) / 4))
        assert [ltc(32, 32, i) for i in range(4)] == [[32]] * 4
        assert all(len(p) >= 1 for p in (ltc(32, 40, i) for i in range(4)))   # 窄区间空档取几何中点

    def test_oa_design_from_config(self, cfg):
        sl = _sl()
        d = [tuple(r) for r in cfg["slicing"]["design"]]
        assert len(d) == 16 and all(len(r) == 3 for r in d)
        for i, j in ((0, 1), (0, 2), (1, 2)):                  # 强度 2：任两列 16 个组合各一次
            assert len({(r[i], r[j]) for r in d}) == 16
        for n in (4, 8, 12, 16):                               # 前缀各轴等频
            for col in range(3):
                counts = [sum(1 for r in d[:n] if r[col] == lv) for lv in range(4)]
                assert counts == [n // 4] * 4
        assert sl.is_orthogonal_array(d) and sl.AutoCTSTierSlicer.from_config(cfg["slicing"]).design == tuple(d)

    def test_stratified_sample(self):
        strat = _sl().stratified_sample
        rng = np.random.default_rng(0)
        y = np.array([0] * 50 + [1] * 30 + [2] * 20)             # 100 样本，比例 5:3:2
        idx = strat(y, n=40, rng=rng, min_per_class=2)
        assert len(idx) == 40 and len(set(idx)) == 40 and list(idx) == sorted(idx)
        counts = np.bincount(y[list(idx)], minlength=3)
        assert counts.tolist() == [20, 12, 8]                      # 目标 = n·类频率 = 40 × (0.5, 0.3, 0.2)，每类 ≥ 2
        idx6 = strat(y, n=6, rng=np.random.default_rng(1), min_per_class=2)
        assert np.bincount(y[list(idx6)], minlength=3).tolist() == [2, 2, 2]
        with pytest.raises(ValueError):
            strat(y, n=5, rng=rng, min_per_class=2)
        assert len(strat(y, n=100, rng=rng, min_per_class=2)) == 100

    def test_prefix_stable_tiers_and_records(self, cfg):
        sl, tc, ltc = _sl(), _sl().tier_candidates, _sl().log_tier_candidates
        rng = np.random.default_rng(0)
        y = rng.integers(0, 5, size=300).astype(np.int64)
        s = sl.AutoCTSTierSlicer.from_config(cfg["slicing"])
        t8 = s.slice(y, n_channels=6, seq_len=200, n_tasks=8, pool_name="Toy")
        t16 = s.slice(y, n_channels=6, seq_len=200, n_tasks=16, pool_name="Toy")
        assert t16[:8] == t8 and len(t16) == 16
        assert [t.task_id for t in t16] == [f"Toy_a{i:02d}" for i in range(16)]
        design = [tuple(r) for r in cfg["slicing"]["design"]]
        n_lo, n_hi = max(32, 2 * 5), min(1024, 300)
        for i, t in enumerate(t16):
            ci, wi, ni = design[i % 16]
            assert t.tag == f"c{ci}w{wi}n{ni}" and t.pool == "Toy"
            assert set(t.classes) == set(range(5))
            assert list(t.channels) == sorted(set(t.channels)) and all(0 <= c < 6 for c in t.channels)
            assert len(t.channels) in tc(1, 6, ci)
            t0, t1 = t.time_window
            assert 0 <= t0 < t1 <= 200 and (t1 - t0) in tc(50, 200, wi)
            n = len(t.sample_indices)
            assert list(t.sample_indices) == sorted(set(t.sample_indices)) and n in ltc(n_lo, n_hi, ni)
            assert np.bincount(y[list(t.sample_indices)], minlength=5).min() >= 2      # 分层：每类 ≥ 2
        assert len({len(t.sample_indices) for t in t16}) > 4                              # 样本数轴真的在变
        t17 = s.slice(y, n_channels=6, seq_len=200, n_tasks=17, pool_name="Toy")
        assert t17[:16] == t16 and t17[16].tag == "c{}w{}n{}".format(*design[0])
        # n_cap = None → 上界 = N；数据集级下限；序列下限
        full = dataclasses.replace(s, n_cap=None)
        top = [t for t in full.slice(y, n_channels=6, seq_len=200, n_tasks=16, pool_name="Toy") if t.tag.endswith("n3")]
        assert all(len(t.sample_indices) in ltc(n_lo, 300, 3) for t in top)
        with pytest.raises(ValueError, match="n_lo"):
            s.slice(y[:20], n_channels=6, seq_len=200, n_tasks=2, pool_name="Toy")
        with pytest.raises(ValueError, match="window_min"):
            s.slice(y, n_channels=6, seq_len=40, n_tasks=2, pool_name="Toy")

    def test_materialize_time_window(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 30, 4)).astype(np.float32)
        y = np.repeat(np.arange(5), 10)
        idx = tuple(range(0, 50, 2))
        t = SlicedTask(task_id="t", pool="p", classes=(0, 1, 2, 3, 4), channels=(1, 3), sample_indices=idx, time_window=(5, 25))
        Xt, yt = materialize_task(t, X, y)
        assert Xt.shape == (25, 20, 2)
        np.testing.assert_array_equal(Xt, X[list(idx)][:, 5:25][:, :, [1, 3]])
        t_old = SlicedTask(task_id="t", pool="p", classes=(0, 1, 2, 3, 4), channels=(1, 3), sample_indices=idx)
        assert t_old.time_window is None and t_old == dataclasses.replace(t, time_window=None)
        assert materialize_task(t_old, X, y)[0].shape == (25, 30, 2)
        assert "time_window" in dataclasses.asdict(t)


# ---------------------------------------------------------------- 候选（PROTOCOL §1.2 / §1.4 锚）
class TestCandidates:
    def test_prune_rules_only_two(self, cfg, dspace, make_cls):
        ac = _ac()
        assert [r["name"] for r in cfg["prune"]["rules"]] == ["tau100", "infonce_tau10"]
        assert ac.is_pruned(make_cls(temperature=100.0), cfg)
        assert ac.is_pruned(make_cls(loss_type="info_nce", temperature=10.0), cfg)
        assert not ac.is_pruned(make_cls(loss_type="triplet", temperature=10.0), cfg)
        assert not ac.is_pruned(make_cls(temperature=0.01), cfg)
        assert not ac.is_pruned(make_cls(norm="none", sim_func="dot", temperature=0.01), cfg)
        assert not ac.is_pruned(make_cls(jitter=0.95, crop=0.95, resize=0.9), cfg)
        assert 0.25 < ac.pruned_share(dspace, cfg, n=4000, seed=0) < 0.34

    def test_shared_catalog(self, aspace, cfg):
        ac = _ac()
        sh = ac.shared_catalog(aspace, cfg)
        assert len(sh) == cfg["candidates"]["n_shared"] == 20
        keys = [k for _c, k in sh]
        assert len(set(keys)) == 20 and all(k == cls_key(c) for c, k in sh)
        assert all(aspace.validate(c) and not ac.is_pruned(c, cfg) for c, _k in sh)
        assert ac.shared_catalog(aspace, cfg, n=10) == sh[:10] and ac.shared_catalog(aspace, cfg) == sh

    def test_random_catalog(self, aspace, cfg):
        ac = _ac()
        excl = {k for _c, k in ac.shared_catalog(aspace, cfg)}
        ra = ac.random_catalog(aspace, cfg, "PhonemeSpectra_a00", exclude_keys=excl)
        rb = ac.random_catalog(aspace, cfg, "PhonemeSpectra_a01", exclude_keys=excl)
        assert len(ra) == len(rb) == cfg["candidates"]["n_random"] == 20
        for r in (ra, rb):
            ks = [k for _c, k in r]
            assert len(set(ks)) == 20 and not (set(ks) & excl)
            assert all(aspace.validate(c) and not ac.is_pruned(c, cfg) for c, _k in r)
        assert ra != rb
        assert ac.random_catalog(aspace, cfg, "PhonemeSpectra_a00", exclude_keys=excl) == ra
        assert ac.random_catalog(aspace, cfg, "PhonemeSpectra_a00", exclude_keys=excl, n=5) == ra[:5]

    def test_aug_max_and_anchor_rule(self, make_cls):
        ac = _ac()
        assert ac.aug_max(make_cls(jitter=0.95, crop=0.5)) == 0.95
        items = [make_cls(jitter=0.95), make_cls(crop=0.8), make_cls(resize=0.9), make_cls(jitter=0.1)]
        sh = [(c, cls_key(c)) for c in items]
        assert ac.pick_anchors(sh, n=2, aug_max_le=0.8) == [sh[1], sh[3]]


# ---------------------------------------------------------------- 标签 / 格（PROTOCOL §1.4）
class TestLabelsAndCells:
    def test_best_of_three_and_floor(self):
        ac = _ac()
        row = {"error": None, "checkpoint_results": [{"acc": 0.4}, {"acc": 0.55}, {"acc": 0.5}]}
        assert ac.best_of_checkpoints(row) == pytest.approx(0.55)
        assert ac.best_of_checkpoints({"error": "boom", "checkpoint_results": None}) is None
        assert ac.floor_of(0.3, [0.4, 0.5, 0.45]) == pytest.approx(0.45)
        assert ac.floor_of(0.6, [0.4, 0.5, 0.45]) == pytest.approx(0.6)
        assert ac.label_of(0.55, 0.45) == (pytest.approx(0.55), False)
        assert ac.label_of(0.40, 0.45) == (pytest.approx(0.45), True)
        assert ac.label_of(0.45, 0.45) == (pytest.approx(0.45), True)
        assert ac.label_of(None, 0.45) == (pytest.approx(0.45), True)

    def test_build_cells_official_options(self, aspace, cfg):
        ac = _ac()
        protos = importlib.import_module("scripts.35_protocol_alignment").load_protocols()
        meta = {"Toy_a00": {"n_train": 100, "T": 60, "C": 2}, "Toy_a01": {"n_train": 700, "T": 200, "C": 5}}
        sh = ac.shared_catalog(aspace, cfg, n=3)
        excl = {k for _c, k in sh}
        randoms = {t: ac.random_catalog(aspace, cfg, t, exclude_keys=excl, n=2) for t in meta}
        anchors = ac.pick_anchors(sh, n=2, aug_max_le=1.0)
        cells = ac.build_cells(meta, sh, randoms, anchors, protos, cfg)
        assert len(cells) == 2 * (3 + 2) + 2 * 2
        keys = [c["key"] for c in cells]
        assert len(set(keys)) == len(keys)
        for c in cells:
            assert c["protocol"] == "ts2vec_official" and c["dataset"] == c["task_id"]
            o = c["options"]
            assert o["batch_size"] == 8 and o["swa"] is True and o["scaler"] == "train" and o["svm_grid"] == "ts2vec" and o["drop_last"] is True
            assert len(c["checkpoints"]) == 3 and c["checkpoints"][-1] == c["iters"]
            assert c["arm"] in ("shared", "random", "anchor")
            assert c["seed"] == (1 if c["arm"] == "anchor" else 0)
            assert aspace.validate(CLS(**c["cls"]))
        assert {c["iters"] for c in cells if c["task_id"] == "Toy_a00"} == {200}
        assert {c["iters"] for c in cells if c["task_id"] == "Toy_a01"} == {600}
        anchor_keys = {k for _c, k in anchors}
        assert {c["cls_key"] for c in cells if c["arm"] == "anchor"} == anchor_keys
        assert [c["task_id"] for c in cells] == sorted([c["task_id"] for c in cells], key=list(meta).index)
        assert ac.estimate_cell_seconds(cells[0], cfg) > 0
        assert ac.estimate_cell_seconds({"iters": 600, "n_train": 700, "T": 200, "C": 5}, cfg) > ac.estimate_cell_seconds({"iters": 200, "n_train": 100, "T": 60, "C": 2}, cfg)

    def test_task_arrays_standardized_on_train(self):
        ac = _ac()
        rng = np.random.default_rng(3)
        X = rng.standard_normal((120, 80, 5)).astype(np.float32)
        X *= np.arange(1, 6, dtype=np.float32)
        X += 10 * np.arange(5, dtype=np.float32)
        y = np.tile(np.arange(4), 30)
        t = SlicedTask(task_id="Toy_a00", pool="Toy", classes=(0, 1, 2, 3), channels=(1, 4),
                       sample_indices=tuple(range(120)), time_window=(10, 70))
        arrays = ac.task_arrays(t, X, y, eval_ratio=0.3, seed=7)
        assert set(arrays) >= {"Xtr", "ytr", "Xte_train", "Xte_split", "yte"}
        Xtr, Xte = arrays["Xtr"], arrays["Xte_train"]
        assert Xtr.shape[1:] == (60, 2) and Xte.shape[1:] == (60, 2)
        assert Xtr.shape[0] + Xte.shape[0] == 120 and Xte.shape[0] == 36
        np.testing.assert_allclose(Xtr.reshape(-1, 2).mean(0), 0.0, atol=1e-4)
        np.testing.assert_allclose(Xtr.reshape(-1, 2).std(0), 1.0, atol=1e-3)
        assert arrays["Xte_split"] is arrays["Xte_train"]
        assert set(arrays["ytr"]) == set(arrays["yte"]) == {0, 1, 2, 3}
        again = ac.task_arrays(t, X, y, eval_ratio=0.3, seed=7)
        np.testing.assert_array_equal(again["ytr"], arrays["ytr"])
        # 评估侧用训练侧统计量：把训练侧统计量套回原尺度应得到原始评估数据
        raw_te = X[:, 10:70][:, :, [1, 4]][arrays["eval_idx"]]
        mu, sd = X[:, 10:70][:, :, [1, 4]][arrays["train_idx"]].reshape(-1, 2).mean(0), X[:, 10:70][:, :, [1, 4]][arrays["train_idx"]].reshape(-1, 2).std(0)
        np.testing.assert_allclose(Xte * sd + mu, raw_te, rtol=1e-4, atol=1e-3)


# ---------------------------------------------------------------- 读数量（PROTOCOL §1.5 / §2）
class TestReadouts:
    def test_split_half_and_interaction_share(self):
        ac = _ac()
        rng = np.random.default_rng(5)
        base = rng.standard_normal(20)
        Z_same = np.stack([base + 0.05 * rng.standard_normal(20) for _ in range(8)])
        Z_diff = rng.standard_normal((8, 20))
        assert ac.split_half_rho(Z_same, half_size=3, n_draws=100, seed=0) > 0.9
        assert ac.split_half_rho(Z_diff, half_size=3, n_draws=100, seed=0) < 0.4
        # 二因素分解：MS_int = SS_int / ((n_t−1)(n_c−1))
        Z = Z_diff
        resid = Z - Z.mean(0, keepdims=True) - Z.mean(1, keepdims=True) + Z.mean()
        ms_int = float((resid ** 2).sum() / (7 * 19))
        m = ac.interaction_share(Z, noise_ms=0.0)
        assert m["ms_int"] == pytest.approx(ms_int) and m["real_share"] == pytest.approx(1.0)
        assert ac.interaction_share(Z, noise_ms=ms_int)["real_share"] == pytest.approx(0.0)
        assert ac.interaction_share(Z, noise_ms=2 * ms_int)["real_share"] < 0
        z0, z1 = np.array([[0.0, 1.0], [2.0, 2.0]]), np.array([[1.0, 1.0], [2.0, 0.0]])
        assert ac.anchor_noise_ms(z0, z1) == pytest.approx((1 + 0 + 0 + 4) / 4 / 2)

    def test_fidelity_metrics(self):
        ac = _ac()
        final = {f"k{i}": 0.3 + 0.02 * i for i in range(20)}
        traj = {"A": {k: (v, v) for k, v in final.items()},                        # 早截点 = 终点
                "B": {k: (1.0 - v, v) for k, v in final.items()}}                    # 完全反向
        augmax = {k: (0.2 if i < 7 else 0.7 if i < 14 else 0.9) for i, k in enumerate(final)}
        m = ac.fidelity_metrics(traj, augmax)
        assert m["rho_per_task"]["A"] == pytest.approx(1.0) and m["rho_per_task"]["B"] == pytest.approx(-1.0)
        assert m["rho_median"] == pytest.approx(0.0)
        assert m["recall3_per_task"]["A"] == pytest.approx(1.0) and m["recall3_per_task"]["B"] == pytest.approx(0.0)
        assert set(m["residual_by_augmax"]) == {"le0.5", "0.5-0.8", "gt0.8"}
        assert m["residual_by_augmax"]["le0.5"]["A"] == pytest.approx(0.0, abs=1e-9)

    def test_champion_retention(self, cfg, make_cls):
        ac = _ac()
        def row(ds, c, acc, seed=0, err=None):
            return {"stage": "s1_catalog", "dataset": ds, "seed": seed, "acc": acc, "error": err, "cls_key": cls_key(c), "cls": c.to_dict()}
        good, bad = make_cls(temperature=1.0), make_cls(temperature=100.0)
        rows = [row("X", good, 0.5), row("X", bad, 0.6), row("Y", good, 0.7), row("Y", bad, 0.6),
                row("Z", good, 0.4), row("Z", bad, 0.9, seed=1), row("Z", bad, None, err="boom")]
        r = ac.champion_retention(rows, cfg)
        assert r["n"] == 3 and r["kept"] == 2 and r["rate"] == pytest.approx(2 / 3)
        assert [d for d, _k in r["lost"]] == ["X"]

    def test_verdict_s1(self):
        v = _ac().verdict_s1
        assert v({"G-A0": True, "G-A1": True, "G-A2": True}) == "ACTS-S1-GO"
        assert v({"G-A0": False, "G-A1": True, "G-A2": True}) == "ACTS-S1-PRUNE"
        assert v({"G-A0": True, "G-A1": False, "G-A2": True}) == "ACTS-S1-SAME"
        assert v({"G-A0": True, "G-A1": True, "G-A2": False}) == "ACTS-S1-NOISE"

    def test_coverage_table(self):
        ac = _ac()
        subsets = [dict(C=2, L=60, n=100, k=4), dict(C=6, L=400, n=700, k=12)]
        targets = {"T1": dict(C=3, L=100, n=200, k=5), "T2": dict(C=64, L=3000, n=278, k=2)}
        t = ac.coverage_table(subsets, targets)
        assert t["per_target"]["T1"]["inside"] == {"C": True, "L": True, "n": True, "k": True}
        assert t["per_target"]["T2"]["inside"]["C"] is False and t["per_target"]["T2"]["inside"]["L"] is False
        assert t["per_target"]["T1"]["nearest_log_dist"] >= 0
        assert t["summary"]["n_targets"] == 2 and t["summary"]["outside_any"] == 1
        assert t["summary"]["outside_by_axis"]["C"] == 1 and t["summary"]["outside_by_axis"]["n"] == 0


# ---------------------------------------------------------------- 部署自检（实验室服务器：npz 重建的划分必须与本地 tasks.json 逐任务一致）
class TestDeploymentCheck:
    def test_split_md5_and_check_deployment(self, cfg, tmp_path):
        g83 = importlib.import_module("scripts.83_acts_corpus")
        ac = _ac()
        rng = np.random.default_rng(0)
        X = rng.standard_normal((120, 80, 5)).astype(np.float32)
        y = np.tile(np.arange(4), 30)
        _tasks, meta = g83.slice_dataset(cfg, "Toy", 2, X=X, y=y)
        assert all(len(m["split_md5"]) == 32 for m in meta.values())
        t0 = next(iter(meta))
        arr = ac.task_arrays(_tasks[0], X, y, eval_ratio=float(cfg["slicing"]["eval_ratio"]), seed=ac.split_seed_for(t0))
        assert ac.split_md5_of(arr) == meta[t0]["split_md5"]
        payload = {"stage": "smoke", "datasets": ["Toy"], "n_tasks": 2, "protocol": cfg["protocol"], "slicing": dict(cfg["slicing"]), "tasks": meta}
        g83._write_json(tmp_path / "tasks.json", payload)
        kw = dict(n_shared=2, n_random=1, anchor_aug_max_le=1.0)
        plan, _catalog = g83.build_plan(cfg, payload, **kw)
        g83._write_json(tmp_path / "plan.json", plan)
        r = g83.check_deployment(cfg, tmp_path, load_dataset=lambda name: (X, y), plan_kwargs=kw)
        assert r["ok"] and r["n_tasks"] == 2 and r["mismatches"] == [] and r["plan_keys_match"] is True
        bad = json.loads((tmp_path / "tasks.json").read_text(encoding="utf-8"))
        bad["tasks"][t0]["split_md5"] = "0" * 32
        g83._write_json(tmp_path / "tasks.json", bad)
        r2 = g83.check_deployment(cfg, tmp_path, load_dataset=lambda name: (X, y), plan_kwargs=kw)
        assert not r2["ok"] and r2["mismatches"][0][0] == t0

