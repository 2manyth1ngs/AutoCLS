"""WP17-ACTS UCR-ADM（results/acts/ucradm/PROTOCOL.md）——先于实现。覆盖：profile ucradm（extends s1pp）、源注册表
configs/acts_sources.yaml 与三级回退、C=1 下 OA 投影为 4×4 全因子、准入 v3′ / chance 护栏 / 决策、多样性读数、
计划 plan_M、读数码、UCR adapter 的 min_class_count 覆盖、命令行入口。零 GPU、不读真实数据。"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

UCR10 = ["NonInvasiveFetalECGThorax1", "CinCECGTorso", "SemgHandMovementCh2", "EOGHorizontalSignal", "ShapesAll",
         "MixedShapesSmallTrain", "LargeKitchenAppliances", "Adiac", "FiftyWords", "FordA"]


def _ac():
    return importlib.import_module("scripts.acts_common")


def _g83():
    return importlib.import_module("scripts.83_acts_corpus")


def _fake_cls(i: int):
    from autocls.space.cls_space import CLS
    return CLS(resize=0.0, rescale=0.0, jitter=0.0, point_mask=0.0, freq_mask=0.0, crop=0.5, order=0, emb_jitter=0.0, emb_mask=0.0,
               norm="none", instance=True, temporal=False, cross_scale=False, kernel=None, pool=None, adj_neighbor=None,
               loss_type="info_nce", sim_func="dot", temperature=[0.01, 0.1, 1.0][i % 3])


@pytest.fixture(scope="module")
def cfg():
    return _ac().load_acts_config(profile="ucradm")


@pytest.fixture(scope="module")
def cfg_pp():
    return _ac().load_acts_config(profile="s1pp")


# ---------------------------------------------------------------- profile（§1.2 / §1.3）
class TestProfile:
    def test_extends_s1pp_slicing_and_catalog_unchanged(self, cfg, cfg_pp):
        assert cfg["profile"] == "ucradm"
        assert cfg["slicing"] == cfg_pp["slicing"] and cfg["eval"] == cfg_pp["eval"]                      # 切片 / cv5 逐位 = s1pp
        assert cfg["prune"] == cfg_pp["prune"] and cfg["candidates"]["n_shared"] == 30                     # shared-30 同一 rng 流 + 同剪枝
        assert cfg["candidates"]["shared_seed"] == cfg_pp["candidates"]["shared_seed"]
        assert cfg["labels"]["shared_seeds"] == [0, 1] and cfg["labels"]["probe0_seeds"] == 3 and cfg["labels"]["floor_mode"] == "diagnostic"

    def test_admission_block(self, cfg):
        a = cfg["admission"]
        assert a["rule"] == "v3" and a["candidates"] == UCR10 and a["task_rows"] == [1, 7, 3, 5]
        assert a["rho_min"] == 0.5 and a["min_pass"] == 3 and a["min_valid_cfg"] == 20
        assert a["guard"] == {"headroom_min": 0.10, "min_pass": 3}
        assert cfg["pilot"]["datasets"] == [] and cfg["pilot"]["n_tasks"] == 8
        assert not cfg.get("whole_eval") and cfg["smoke"]["dataset"] == "Adiac"
        assert cfg["verdict"] == {"go_min": 7, "partial_min": 4, "diverse_rho_max": 0.3, "diverse_min": 2}


# ---------------------------------------------------------------- 源注册表（§1.1）+ 铁律 6
class TestSourcesRegistry:
    def test_registry_lists_ten_npz_univariate(self):
        ac = _ac()
        reg = ac.acts_sources_registry()
        names = [e.name for e in reg.entries]
        assert names == UCR10
        assert all(e.source == "npz" and e.n_channels == 1 and e.role == "source" for e in reg.entries)

    def test_no_name_clash_with_targets_or_old_sources(self):
        ac = _ac()
        from autocls.data.source_pool import load_registry
        targets = set(ac.target_sizes_29()) | {"UCI-HAR", "Epilepsy", "FaceDetection"}
        old = {e.name for e in load_registry().entries}
        assert not (set(UCR10) & targets) and not (set(UCR10) & old)

    def test_load_dataset_full_third_level_fallback(self, monkeypatch):
        import autocls.data.source_pool as sp
        ac = _ac()
        seen = []
        monkeypatch.setattr(sp, "load_source_dataset", lambda name, reg, crop_len="auto": (seen.append((name, reg.get(name).source, crop_len)) or (np.zeros((4, 8, 1), np.float32), np.arange(4))))
        X, y = ac.load_dataset_full("Adiac")                       # 只在 acts_sources.yaml
        assert seen == [("Adiac", "npz", None)] and X.shape == (4, 8, 1)
        ac.load_dataset_full("PTB-XL")                             # 仍走 candidate_pool
        assert seen[-1] == ("PTB-XL", "npz", None)
        with pytest.raises(KeyError):
            ac.load_dataset_full("NoSuchDataset")


# ---------------------------------------------------------------- C = 1 的 OA 退化（§1.2）
class TestUnivariateDesign:
    def test_oa_projection_is_full_factorial(self):
        from autocls.data.acts_slicer import oa16_design
        pairs = {(b, n) for _a, b, n in oa16_design()}
        assert len(pairs) == 16 and pairs == {(b, n) for b in range(4) for n in range(4)}

    def test_slicer_single_channel_16_distinct_window_n_tiers(self, cfg):
        from autocls.data.acts_slicer import AutoCTSTierSlicer
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, size=600)
        sl = AutoCTSTierSlicer.from_config(cfg["slicing"])
        tasks = sl.slice(y, n_channels=1, seq_len=300, n_tasks=16, pool_name="U")
        assert all(t.channels == (0,) for t in tasks)
        tags = {(t.tag[3], t.tag[5]) for t in tasks}                # "c{ci}w{wi}n{ni}" → (wi, ni)
        assert len(tags) == 16
        rows = [1, 7, 3, 5]
        assert [(tasks[r].tag[3], tasks[r].tag[5]) for r in rows] == [("2", "3"), ("0", "3"), ("1", "2"), ("3", "2")]


# ---------------------------------------------------------------- 准入 v3′ + 护栏（§1.4 / §1.5）
class TestAdmissionV3:
    def _tasks(self, rho_good: bool, n_tasks: int = 4, n_cfg: int = 30, seed: int = 0):
        rng = np.random.default_rng(seed)
        out = {}
        for t in range(n_tasks):
            base = np.sort(rng.uniform(0.3, 0.9, size=n_cfg))[::-1]
            a0 = base + rng.normal(scale=0.005, size=n_cfg)
            a1 = (base + rng.normal(scale=0.005, size=n_cfg)) if rho_good else rng.uniform(0.3, 0.9, size=n_cfg)
            out[f"S_a{t:02d}"] = (a0.tolist(), a1.tolist())
        return out

    def test_metrics_pass_and_fail(self, cfg):
        ac = _ac()
        m = ac.admission_v3_metrics(self._tasks(True), rho_min=0.5, min_valid_cfg=20)
        assert m["n"] == 4 and m["pass"] == 4 and all(r > 0.9 for r in m["rho"])
        m2 = ac.admission_v3_metrics(self._tasks(False), rho_min=0.5, min_valid_cfg=20)
        assert m2["n"] == 4 and m2["pass"] <= 1

    def test_metrics_missing_handling(self, cfg):
        ac = _ac()
        t = self._tasks(True)
        a0, a1 = t["S_a00"]
        a1 = list(a1); a1[3] = None; a1[7] = float("nan")             # 单种子缺 → 该配置剔除
        t["S_a00"] = (a0, a1)
        a0b, a1b = t["S_a01"]
        t["S_a01"] = (a0b[:15] + [None] * 15, a1b)                      # 齐的配置 < 20 → 该任务不计
        m = ac.admission_v3_metrics(t, rho_min=0.5, min_valid_cfg=20)
        assert m["n"] == 3 and m["n_valid"]["S_a00"] == 28 and "S_a01" in m["skipped"]

    def test_guard_and_decision(self, cfg):
        ac = _ac()
        t = self._tasks(True)
        maj = {tid: 0.2 for tid in t}
        g = ac.chance_guard(t, maj, headroom_min=0.10)
        assert g["n"] == 4 and g["pass"] == 4 and all(h > 0.10 for h in g["h"])
        g2 = ac.chance_guard(t, {tid: 0.85 for tid in t}, headroom_min=0.10)
        assert g2["pass"] == 0
        metrics = {"A": {**ac.admission_v3_metrics(t, rho_min=0.5, min_valid_cfg=20), "guard": g},
                   "B": {**ac.admission_v3_metrics(self._tasks(False), rho_min=0.5, min_valid_cfg=20), "guard": g},
                   "C": {**ac.admission_v3_metrics(t, rho_min=0.5, min_valid_cfg=20), "guard": g2},
                   "D": {**ac.admission_v3_metrics(t, rho_min=0.5, min_valid_cfg=20), "guard": g, "n": 2, "pass": 2}}
        d = ac.admission_v3_decision(metrics, cfg)
        assert d["admitted"] == ["A"] and d["rule"] == "v3"
        assert "rho" in d["reasons"]["B"] and "guard" in d["reasons"]["C"] and "na" in d["reasons"]["D"]

    def test_decision_strict_majority_of_four(self, cfg):
        ac = _ac()
        g = {"n": 4, "pass": 4, "h": [0.3] * 4}
        mk = lambda p: {"n": 4, "pass": p, "rho": [0.9] * p + [0.1] * (4 - p), "guard": g}
        d = ac.admission_v3_decision({"P3": mk(3), "P2": mk(2)}, cfg)
        assert d["admitted"] == ["P3"]


# ---------------------------------------------------------------- 多样性读数（§1.7）
class TestDiversity:
    def test_profile_rho_and_pairwise(self):
        ac = _ac()
        ref = np.linspace(1, -1, 30)
        prof = {"same": ref + 0.01, "anti": -ref, "flat": np.r_[np.linspace(1, -1, 15), np.linspace(1, -1, 15)]}
        r = ac.profile_diversity(prof, ref)
        assert r["rho_vs_ref"]["same"] > 0.99 and r["rho_vs_ref"]["anti"] < -0.99
        M = np.array(r["pairwise"]); assert M.shape == (3, 3) and M[0, 1] == pytest.approx(-1.0, abs=1e-6) and np.allclose(np.diag(M), 1.0)
        assert r["names"] == ["same", "anti", "flat"]

    def test_between_source_decomp(self):
        ac = _ac()
        rng = np.random.default_rng(0)
        base = np.linspace(0.9, 0.5, 30)
        same = {s: np.tile(base, (4, 1)) + rng.normal(scale=0.01, size=(4, 30)) for s in ("A", "B", "C")}   # 噪声 ≈ 相邻档距，秩有抖动但序一致
        r = ac.between_source_decomp(same)
        assert r["ss_between"] + r["ss_within"] == pytest.approx(1.0) and r["ss_between"] < 0.2
        assert all(v > 0.9 for v in r["agree"].values())
        diff = {**same, "D": np.tile(base[::-1], (4, 1)) + rng.normal(scale=0.01, size=(4, 30))}
        r2 = ac.between_source_decomp(diff)
        assert r2["ss_between"] > r["ss_between"] and r2["agree"]["D"] < 0


# ---------------------------------------------------------------- 计划 plan_M（§1.3）+ 读数码（§1.8）
class TestPlanAndVerdict:
    def _meta(self, srcs):
        meta = {}
        for s in srcs:
            for r in range(8):
                meta[f"{s}_a{r:02d}"] = {"dataset": s, "n_train": 200, "T": 100, "C": 1, "k": 5, "majority": 0.3, "iters": 200,
                                         "tag": f"c0w{r % 4}n{r % 4}", "time_window": [0, 100], "fold_md5": "x"}
        return meta

    def test_build_plan_adm_counts(self, cfg):
        g = _g83()
        srcs = ["Adiac", "FordA"]
        c2 = {**cfg, "admission": {**cfg["admission"], "candidates": srcs}}
        shared = [(_fake_cls(i), f"key{i:02d}") for i in range(30)]
        plan = g.build_plan_adm(c2, self._meta(srcs), shared=shared, existing_keys=set())
        cells = plan["cells"]
        assert plan["phase"] == "M" and plan["n_cells"] == 2 * 4 * 30 * 2 and len({c["key"] for c in cells}) == plan["n_cells"]
        assert all(c["arm"] == "shared" and c["eval"] == "cv5" for c in cells) and {c["seed"] for c in cells} == {0, 1}
        assert plan["probe_tasks"] == ["Adiac_a01", "Adiac_a07", "Adiac_a03", "Adiac_a05", "FordA_a01", "FordA_a07", "FordA_a03", "FordA_a05"]
        assert plan["n_probe"] == 8 * 3 and plan["n_new"] == plan["n_cells"]
        done = {c["key"] for c in cells[:50]}
        assert g.build_plan_adm(c2, self._meta(srcs), shared=shared, existing_keys=done)["n_new"] == plan["n_cells"] - 50

    def test_verdict(self, cfg):
        g = _g83()
        assert g.verdict_ucradm(7, 2, cfg) == "UCR-ADM-GO+DIVERSE"
        assert g.verdict_ucradm(9, 1, cfg) == "UCR-ADM-GO+SAME"
        assert g.verdict_ucradm(5, 3, cfg) == "UCR-ADM-PARTIAL+DIVERSE"
        assert g.verdict_ucradm(3, 0, cfg) == "UCR-ADM-DEAD+SAME"

    def test_parser_accepts_new_flags(self):
        g = _g83()
        a = g.build_parser().parse_args(["--plan-adm", "--analyze-adm", "--phase", "M", "--profile", "ucradm"])
        assert a.plan_adm and a.analyze_adm and a.phase == "M"


# ---------------------------------------------------------------- UCR adapter：min_class_count 覆盖（§1.1）
class TestUcrAdapters:
    NEW = ["ucr_cincecgtorso", "ucr_semghandmovementch2", "ucr_eoghorizontalsignal", "ucr_shapesall",
           "ucr_mixedshapessmalltrain", "ucr_largekitchenappliances", "ucr_adiac", "ucr_fiftywords"]

    def test_specs(self):
        names = []
        for mod in self.NEW:
            spec = importlib.import_module(f"scripts.candidates.{mod}").SPEC
            assert spec["segmented"] is True and spec["min_class_count"] == 5
            names.append(spec["name"])
        assert names == ["CinCECGTorso", "SemgHandMovementCh2", "EOGHorizontalSignal", "ShapesAll", "MixedShapesSmallTrain",
                         "LargeKitchenAppliances", "Adiac", "FiftyWords"]

    def test_effective_min_class_count(self):
        pc = importlib.import_module("scripts.prepare_candidates")
        assert pc.effective_min_class_count({"name": "x"}) == pc.MIN_CLASS_COUNT == 64
        assert pc.effective_min_class_count({"name": "x", "min_class_count": 5}) == 5
