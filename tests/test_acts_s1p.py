"""WP17-ACTS S1′（results/acts/s1p/PROTOCOL.md）——先于实现。覆盖：profile 深合并（基线不动）、样本数轴 n_lo = max(80, 7k) 与每类 ≥ 5、
分层 5 折 CV 切片数组、CV 执行体（终点标签、5 折均值）、39.execute 的 runner 注入、阶段 A / B 计划（准入块 / 3× 消融 / 主批去重）、
终点 + 两种子均值标签、准入 / 预算 / 噪声读数量、判定规则。零 GPU、不读真实数据。"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from autocls.data.task_slicer import SlicedTask


def _ac():
    return importlib.import_module("scripts.acts_common")


def _sl():
    return importlib.import_module("autocls.data.acts_slicer")


def _g83():
    return importlib.import_module("scripts.83_acts_corpus")


@pytest.fixture(scope="module")
def cfg():
    return _ac().load_acts_config(profile="s1p")


@pytest.fixture(scope="module")
def base():
    return _ac().load_acts_config()


# ---------------------------------------------------------------- profile / 切片（§1.1）
class TestProfileAndSlicing:
    def test_profile_deep_merge_keeps_base(self, cfg, base):
        assert cfg["profile"] == "s1p" and "profile" not in base
        assert cfg["slicing"]["n_min"] == 80 and cfg["slicing"]["n_lo_per_class"] == 7 and cfg["slicing"]["n_min_per_class"] == 5
        assert cfg["slicing"]["design"] == base["slicing"]["design"] and cfg["slicing"]["window_min"] == 50   # 未覆盖的键沿用基线
        assert base["slicing"]["n_min"] == 32 and base["slicing"]["n_min_per_class"] == 2 and "n_lo_per_class" not in base["slicing"]
        assert cfg["labels"]["aggregate"] == "endpoint" and cfg["labels"]["shared_seeds"] == [0, 1] and cfg["labels"]["anchors"] == "none"
        assert cfg["prune"] == base["prune"] and cfg["eval"]["mode"] == "cv5"

    def test_sample_bounds_7k(self, cfg, base):
        sl = _sl()
        s = sl.AutoCTSTierSlicer.from_config(cfg["slicing"])
        assert s.sample_bounds(6668, 39) == (273, 1024) and s.sample_bounds(180, 12) == (84, 180) and s.sample_bounds(440, 8) == (80, 440)
        assert s.sample_bounds(1000, 26) == (182, 1000) and s.n_min_per_class == 5
        s0 = sl.AutoCTSTierSlicer.from_config(base["slicing"])
        assert s0.sample_bounds(6668, 39) == (78, 1024)                     # S1 基线逐位不变（n_lo_per_class 缺省 = n_min_per_class）

    def test_slice_min_per_class_5(self, cfg):
        sl = _sl()
        rng = np.random.default_rng(0)
        y = rng.integers(0, 5, size=300).astype(np.int64)
        s = sl.AutoCTSTierSlicer.from_config(cfg["slicing"])
        for t in s.slice(y, n_channels=6, seq_len=200, n_tasks=8, pool_name="Toy"):
            counts = np.bincount(y[list(t.sample_indices)], minlength=5)
            assert counts.min() >= 5 and len(t.sample_indices) >= 80


# ---------------------------------------------------------------- CV 切片数组 / 执行体（§1.2）
class TestCV:
    def _task(self):
        return SlicedTask(task_id="Toy_a01", pool="Toy", classes=(0, 1, 2, 3), channels=(1, 4), sample_indices=tuple(range(120)), time_window=(10, 70))

    def test_task_arrays_cv(self):
        ac = _ac()
        rng = np.random.default_rng(3)
        X = (rng.standard_normal((120, 80, 5)) * np.arange(1, 6) + 10 * np.arange(5)).astype(np.float32)
        y = np.tile(np.arange(4), 30)
        a = ac.task_arrays_cv(self._task(), X, y, n_folds=5, seed=7)
        assert set(a) >= {"X", "y", "folds", "fold_md5", "majority"}
        assert a["X"].shape == (120, 60, 2) and a["X"].dtype == np.float32
        np.testing.assert_allclose(a["X"].reshape(-1, 2).mean(0), 0.0, atol=1e-4)
        np.testing.assert_allclose(a["X"].reshape(-1, 2).std(0), 1.0, atol=1e-3)
        assert len(a["folds"]) == 5
        allte = np.sort(np.concatenate([te for _tr, te in a["folds"]]))
        np.testing.assert_array_equal(allte, np.arange(120))                  # 测试折互斥且覆盖全部
        for tr, te in a["folds"]:
            assert set(a["y"][tr]) == {0, 1, 2, 3} and not (set(tr) & set(te))
        b = ac.task_arrays_cv(self._task(), X, y, n_folds=5, seed=7)
        assert b["fold_md5"] == a["fold_md5"] and ac.task_arrays_cv(self._task(), X, y, n_folds=5, seed=8)["fold_md5"] != a["fold_md5"]
        assert a["majority"] == pytest.approx(0.25)

    def test_cv_eval_on_separable(self):
        ac = _ac()
        rng = np.random.default_rng(0)
        y = np.repeat(np.arange(3), 40)
        feats = (rng.standard_normal((120, 6)) + y[:, None] * 8.0).astype(np.float32)
        folds = ac.stratified_folds(y, n_folds=5, seed=1)
        v = ac.cv_eval(feats, y, folds, grid="ts2vec", seed=0)
        assert v["acc"] == pytest.approx(1.0) and len(v["fold_acc"]) == 5 and v["f1"] == pytest.approx(1.0)

    def test_run_trajectory_cell_cv(self, monkeypatch):
        g83 = _g83()
        rng = np.random.default_rng(1)
        arrays = {"X": rng.standard_normal((40, 16, 2)).astype(np.float32), "y": np.tile(np.arange(2), 20),
                  "folds": _ac().stratified_folds(np.tile(np.arange(2), 20), n_folds=5, seed=0), "majority": 0.5}
        calls = []

        def fake_pretrain(enc, cls, X, *, max_iters, checkpoints, on_checkpoint, **kw):
            calls.append((tuple(X.shape), max_iters, tuple(checkpoints)))
            for step in checkpoints:
                on_checkpoint(step, enc)
            return [1.0, 0.5]

        monkeypatch.setattr(g83, "pretrain_encoder", fake_pretrain)
        monkeypatch.setattr(g83, "encode_features", lambda module, X, device=None: X.mean(1))
        seq = iter([0.3, 0.6, 0.9])
        monkeypatch.setattr(g83.ac, "cv_eval", lambda feats, y, folds, grid, seed: {"acc": next(seq), "f1": 0.1, "fold_acc": [0.0] * 5, "svm_c": [1.0] * 5, "svm_cv_score": [None] * 5})
        cell = {"key": "k", "dataset": "Toy_a01", "task_id": "Toy_a01", "arm": "shared", "cls_key": "abc", "seed": 0, "iters": 12, "checkpoints": [3, 6, 12],
                "options": {"batch_size": 8, "lr": 1e-3, "drop_last": True, "swa": True, "crop_mode": "fixed", "scaler": "train", "svm_grid": "ts2vec", "input_mask": "none", "repr_dropout": 0.0, "engine": "cl_trainer"},
                "cls": {k: v for k, v in _base_cls().items()}}
        row = g83.run_trajectory_cell_cv(cell, arrays, "cpu")
        assert row["error"] is None and row["eval"] == "cv5" and row["acc"] == pytest.approx(0.9)
        assert [c["acc"] for c in row["checkpoint_results"]] == pytest.approx([0.3, 0.6, 0.9]) and row["n_train"] == 40
        assert calls[0] == ((40, 16, 2), 12, (3, 6, 12))

    def test_execute_runner_injection(self, monkeypatch, tmp_path):
        import types
        g39 = importlib.import_module("scripts.39_sh_deploy")
        seen = []
        cells = [{"key": "a", "dataset": "D1", "arm": "shared", "cls_key": "k1234567", "seed": 0, "iters": 8, "checkpoints": [8]}]
        args = types.SimpleNamespace(outdir=str(tmp_path), resume=False, joblib_backend="threading", device="cpu")
        g39.execute(cells, args, "t", loader=lambda d, npz=None: {"X": None},
                    runner=lambda cell, arrays, device: (seen.append(cell["key"]) or {**cell, "acc": 0.5, "error": None, "checkpoint_results": [{"acc": 0.5}], "pretrain_seconds": 1, "eval_seconds": 0.1}))
        assert seen == ["a"] and len((tmp_path / "runs.jsonl").read_text(encoding="utf-8").strip().splitlines()) == 1


def _base_cls():
    return dict(resize=0.0, rescale=0.0, jitter=0.3, point_mask=0.0, freq_mask=0.0, crop=0.5, order=0, emb_jitter=0.0, emb_mask=0.0, norm="none",
                instance=True, temporal=False, cross_scale=False, kernel=None, pool=None, adj_neighbor=None, loss_type="info_nce", sim_func="cosine", temperature=0.1)


# ---------------------------------------------------------------- 计划（§1.4 / §1.5 / §1.7）
def _toy_meta(name: str, n_tasks: int = 8) -> dict:
    design = _ac().load_acts_config()["slicing"]["design"]
    return {f"{name}_a{i:02d}": {"dataset": name, "n_train": 100, "n_eval": 0, "T": 60, "C": 2, "k": 4, "majority": 0.3, "iters": 200,
                                 "tag": "c{}w{}n{}".format(*design[i]), "time_window": [0, 60]} for i in range(n_tasks)}


class TestPlanS1P:
    def test_phase_a_and_b(self, cfg):
        g83 = _g83()
        meta = {**_toy_meta("Cricket"), **_toy_meta("Other")}
        c2 = {**cfg, "admission": {**cfg["admission"], "candidates": ["Cricket", "Other"]}, "pilot": {**cfg["pilot"], "datasets": ["Cricket", "Other"], "add_if_admitted": []}}
        plan_a, catalog = g83.build_plan_s1p(c2, meta, phase="A", n_shared=3, n_random=2)
        cells = plan_a["cells"]
        adm = [c for c in cells if c["arm"] == "shared"]
        abl = [c for c in cells if c["arm"] == "ablation3x"]
        assert len(adm) == 2 * 2 * 3 * 2 and len(abl) == 8 * 3 and len(cells) == len(adm) + len(abl)
        assert {c["task_id"][-3:] for c in adm} == {"a01", "a07"} and {c["seed"] for c in adm} == {0, 1}
        assert all(c["iters"] == 600 and len(c["checkpoints"]) == 5 and c["checkpoints"][-1] == 600 and c["checkpoints"][2] == 200 and c["seed"] == 0 for c in abl)
        assert all(c["task_id"].startswith("Cricket") for c in abl)
        keys = [c["key"] for c in cells]
        assert len(set(keys)) == len(keys) and all(c["eval"] == "cv5" for c in cells)
        plan_b, _ = g83.build_plan_s1p(c2, meta, phase="B", n_shared=3, n_random=2, admitted=["Cricket", "Other"], skip_keys=set(keys))
        cells_b = plan_b["cells"]
        assert all(c["arm"] in ("shared", "random") for c in cells_b) and not ({c["key"] for c in cells_b} & set(keys))
        assert len(cells_b) == 2 * 8 * (3 * 2 + 2) - len(adm)
        assert plan_b["n_probe"] == 2 * 8 * 3
        assert list(catalog["random"]) == list(meta) and len(catalog["shared"]) == 3

    def test_pilot_sets_rule(self, cfg):
        g83 = _g83()
        assert g83.pilot_sets(cfg, admitted={"PhonemeSpectra", "Cricket", "UWaveGestureLibrary", "HandMovementDirection"}) == ["PhonemeSpectra", "Cricket", "UWaveGestureLibrary", "HandMovementDirection"]
        assert g83.pilot_sets(cfg, admitted={"PhonemeSpectra", "Cricket", "UWaveGestureLibrary"}) == ["PhonemeSpectra", "Cricket", "UWaveGestureLibrary"]


# ---------------------------------------------------------------- 标签 / 读数量（§1.3 / §2）
class TestReadoutsS1P:
    def test_endpoint_two_seed_mean_label(self):
        ac = _ac()
        rows = {0: {"error": None, "checkpoint_results": [{"acc": 0.9}, {"acc": 0.5}, {"acc": 0.6}]},
                1: {"error": None, "checkpoint_results": [{"acc": 0.1}, {"acc": 0.2}, {"acc": 0.7}]}}
        assert ac.endpoint_of(rows[0]) == pytest.approx(0.6)                              # 终点不是 best-of-3
        assert ac.endpoint_of({"error": "x", "checkpoint_results": None}) is None
        r, failed = ac.label_multi_seed([rows[0], rows[1]], floor=0.55)
        assert r == pytest.approx(0.65) and failed is False
        r2, f2 = ac.label_multi_seed([rows[0], {"error": "boom"}], floor=0.55)
        assert r2 == pytest.approx(0.55) and f2 is True                                     # 任一种子缺 → 地板
        assert ac.label_multi_seed([rows[0]], floor=0.6) == (pytest.approx(0.6), True)

    def test_noise_of_mean_and_admission_rho(self):
        ac = _ac()
        z0, z1 = np.array([[0.0, 1.0], [2.0, 2.0]]), np.array([[1.0, 1.0], [2.0, 0.0]])
        assert ac.noise_ms_of_mean(z0, z1) == pytest.approx((1 + 0 + 0 + 4) / 4 / 4)
        s0 = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.55, 0.65, 0.75, 0.85, 0.95])
        s1 = s0 + 0.01 * np.arange(10)
        f0 = np.zeros(10, bool); f1 = np.zeros(10, bool); f1[3] = True
        m = ac.admission_rho(s0, s1, f0, f1, min_survivors=8)
        assert m["n_survivors"] == 9 and m["rho"] > 0.9
        f1[:5] = True
        assert np.isnan(ac.admission_rho(s0, s1, f0, f1, min_survivors=8)["rho"])

    def test_admission_and_budget_decisions(self, cfg):
        g83 = _g83()
        metrics = {"Cricket": 0.8, "Ethanol": 0.2, "HMD": float("nan"), "SCP1": 0.5, "A": 0.4, "B": 0.4, "C": 0.4, "D": 0.4}
        c2 = {**cfg, "admission": {**cfg["admission"], "candidates": list(metrics), "fallback": ["SCP1"], "pool_floor": 7}}
        d = g83.admission_decision(metrics, c2)
        assert d["admitted"] == ["Cricket", "A", "B", "C", "D"] and d["rejected"] == ["Ethanol", "HMD"] and d["fallback_used"] == ["SCP1"] and d["pool"] == ["Cricket", "A", "B", "C", "D", "SCP1"]
        assert d["below_floor"] is True
        assert g83.budget_decision({"failed_drop": 0.2, "surv_rho_13_median": 0.3}, cfg) == "3x"
        assert g83.budget_decision({"failed_drop": 0.2, "surv_rho_13_median": 0.6}, cfg) == "1x"
        assert g83.budget_decision({"failed_drop": 0.1, "surv_rho_13_median": 0.3}, cfg) == "1x"

    def test_ablation_readout(self):
        g83 = _g83()
        cfgs = [f"k{i}" for i in range(10)]
        base = np.linspace(0.4, 0.9, 10)
        rows = [{"task_id": "Cricket_a00", "cls_key": k, "error": None,
                 "checkpoint_results": [{"acc": v - 0.3}, {"acc": v - 0.2}, {"acc": v - 0.1}, {"acc": v}, {"acc": v + 0.05}]} for k, v in zip(cfgs, base)]
        r = g83.ablation_readout(rows, floors={"Cricket_a00": 0.45})
        t = r["per_task"]["Cricket_a00"]
        assert t["failed_1x"] == pytest.approx(0.5) and t["failed_3x"] == pytest.approx(0.0)      # 1× 终点 = 第 3 截点
        assert t["surv_rho_13"] == pytest.approx(1.0) and t["surv_rho_23"] == pytest.approx(1.0)
        assert r["failed_drop"] == pytest.approx(0.5) and r["surv_rho_13_median"] == pytest.approx(1.0)

    def test_verdict_s1p(self):
        v = _g83().verdict_s1p
        assert v({"G-A1": True, "G-A2": True}) == "ACTS-S1P-GO" and v({"G-A1": False, "G-A2": True}) == "ACTS-S1P-SAME" and v({"G-A1": True, "G-A2": False}) == "ACTS-S1P-NOISE"
