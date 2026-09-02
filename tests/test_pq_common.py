"""MX-3-PQ 公共模块（scripts/pq_common.py）纯逻辑测试。

协议：.superpowers/sdd/autocts-e-autocts-autocts-e-autocts-refactored-moler/
interfaces.md「Budgets / ids」「Task 4 → Tasks 5, 6, 7」。

本文件**绝不加载数据集**（aeon / UEA 缓存不参与）——PQDataset 装载只在 smoke/pod
路径跑；此处只覆盖纯装配逻辑，期望值手工推导。δ 公式与 P95 索引规则逐字对齐
scripts/2c_generate_labels_workpoint.py::calibrate 与 scripts/7_swa_noise_probe.py::_p95。
"""
import dataclasses
import json

import numpy as np
import pytest

from autocls.data.task_slicer import SlicedTask
from autocls.labeling import cls_key, load_probe_cls, run_seed_for
from autocls.space.cls_encoder import CLSEncoder
from autocls.space.cls_space import CLSSpace
from scripts import pq_common as pq


# --------------------------------------------------------------- 常量
class TestConstants:
    def test_pool_shape(self):
        assert len(pq.PQ_DATASETS) == 8
        assert pq.TRAINING_ONLY == ("SelfRegulationSCP2",)
        assert len(pq.FOLD_TARGETS) == 7
        assert "SelfRegulationSCP2" not in pq.FOLD_TARGETS
        assert set(pq.FOLD_TARGETS) | set(pq.TRAINING_ONLY) == set(pq.PQ_DATASETS)

    def test_budgets_and_slices(self):
        assert pq.BUDGETS == (60, 120, 290)
        assert pq.N_SLICES == 6

    def test_reg_of_dual_registry(self):
        # 25_mx2_endtoend:72 同款双注册表：UniMiB 在 candidate_pool，其余默认
        assert pq.REG_OF["UniMiB-SHAR"] == "configs/candidate_pool.yaml"
        assert pq.REG_OF["Handwriting"] is None
        assert set(pq.REG_OF) == set(pq.PQ_DATASETS)


# --------------------------------------------------------------- id
class TestEngineTaskIds:
    def test_engine_task_id(self):
        assert pq.engine_task_id("Handwriting_03", 120) == "Handwriting_03@120"

    def test_round_trip(self):
        for slc in ("Handwriting_03", "UniMiB-SHAR_00", "SelfRegulationSCP2_02"):
            for b in pq.BUDGETS:
                tid = pq.engine_task_id(slc, b)
                assert pq.split_engine_task_id(tid) == (slc, b)

    def test_split_returns_int_budget(self):
        assert pq.split_engine_task_id("Cricket_01@60")[1] == 60
        assert isinstance(pq.split_engine_task_id("Cricket_01@60")[1], int)

    def test_split_rejects_plain_slice(self):
        with pytest.raises(ValueError):
            pq.split_engine_task_id("Cricket_01")

    def test_dataset_of_both_forms(self):
        assert pq.dataset_of("Handwriting_03") == "Handwriting"
        assert pq.dataset_of("Handwriting_03@120") == "Handwriting"
        assert pq.dataset_of("UniMiB-SHAR_00@60") == "UniMiB-SHAR"
        assert pq.dataset_of("UniMiB-SHAR_00") == "UniMiB-SHAR"
        assert pq.dataset_of("SelfRegulationSCP2_02@290") == "SelfRegulationSCP2"

    def test_slice_of(self):
        assert pq.slice_of("Handwriting_03@120") == "Handwriting_03"
        assert pq.slice_of("Handwriting_03") == "Handwriting_03"


# --------------------------------------------------------------- ctx
class TestCtx:
    def test_one_hot_per_budget(self):
        for i, b in enumerate(pq.BUDGETS):
            v = pq.ctx_vector(b)
            assert v.dtype == np.float32
            assert v.shape == (3,)
            assert v.sum() == pytest.approx(1.0)
            assert v[i] == pytest.approx(1.0)

    def test_unknown_budget_raises(self):
        with pytest.raises(ValueError):
            pq.ctx_vector(999)

    def test_ctx_dim(self):
        assert pq.ctx_dim(pq.BUDGETS) == 3
        assert pq.ctx_dim((60, 120)) == 2


# --------------------------------------------------------------- 折划分
class TestFoldSplit:
    def test_all_seven_targets(self):
        assert len(pq.FOLD_TARGETS) == 7
        for target in pq.FOLD_TARGETS:
            train, test = pq.fold_split(target)
            assert test == target
            assert target not in train
            assert len(train) == 7
            assert "SelfRegulationSCP2" in train       # 低权 EEG 永远在训练侧
            assert set(train) | {target} == set(pq.PQ_DATASETS)

    def test_training_only_cannot_be_target(self):
        with pytest.raises(ValueError):
            pq.fold_split("SelfRegulationSCP2")

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError):
            pq.fold_split("NotADataset")


# --------------------------------------------------------------- S2 划分
class TestS2Split:
    def _synthetic(self):
        out = []
        for d in ("Handwriting", "Cricket"):
            out += [f"{d}_{i:02d}" for i in range(6)]
        out += [f"SelfRegulationSCP2_{i:02d}" for i in range(3)]
        return out

    def test_split_positions(self):
        train, test = pq.s2_split(self._synthetic())
        for d in ("Handwriting", "Cricket"):
            assert [s for s in train if s.startswith(d)] == [f"{d}_0{i}" for i in range(4)]
            assert [s for s in test if s.startswith(d)] == [f"{d}_04", f"{d}_05"]
        assert [s for s in train if s.startswith("SelfRegulationSCP2")] == [
            "SelfRegulationSCP2_00", "SelfRegulationSCP2_01"]
        assert [s for s in test if s.startswith("SelfRegulationSCP2")] == [
            "SelfRegulationSCP2_02"]

    def test_disjoint_and_complete(self):
        allsl = self._synthetic()
        train, test = pq.s2_split(allsl)
        assert set(train) & set(test) == set()
        assert set(train) | set(test) == set(allsl)

    def test_accepts_mapping(self):
        mapping = {"Cricket": [f"Cricket_{i:02d}" for i in range(6)]}
        train, test = pq.s2_split(mapping)
        assert test == ["Cricket_04", "Cricket_05"]

    def test_unexpected_slice_count_raises(self):
        with pytest.raises(ValueError):
            pq.s2_split(["Cricket_00", "Cricket_01", "Cricket_02", "Cricket_03"])


# --------------------------------------------------------------- 记录装配
@pytest.fixture
def synth_cache(make_cls):
    """load_rprime_cache 形状的合成缓存：3 切片 × 2 CLS × 预算 {60, 120}。"""
    cache = {}
    cls_by_key = {}
    for j, jit in enumerate((0.1, 0.3)):
        c = make_cls(jitter=jit)
        cls_by_key[cls_key(c)] = (j, c)
    for tid in ("Handwriting_00", "Handwriting_01", "UniMiB-SHAR_00"):
        for b in (60, 120):
            for key, (j, c) in cls_by_key.items():
                cache[(tid, key, b)] = {
                    "task_id": tid, "cls_key": key, "iters": b,
                    "origin": "shared" if j == 0 else "random",
                    "r_prime": 0.5 + 0.1 * j + b / 10000.0,
                    "n_eval": 300, "seed": 0,
                    "cls": dataclasses.asdict(c),
                }
    return cache


class TestBuildRecords:
    def test_count_and_rewrite(self, synth_cache):
        enc = CLSEncoder(CLSSpace.from_yaml())
        slices = ["Handwriting_00", "Handwriting_01"]
        recs = pq.build_records(synth_cache, enc, budgets=pq.BUDGETS, slices=slices)
        # 2 切片 × 2 预算（290 缓存里没有）× 2 CLS = 8
        assert len(recs) == 8
        assert {r.task_id for r in recs} == {
            f"{s}@{b}" for s in slices for b in (60, 120)}
        assert all(r.vec.ndim == 1 for r in recs)

    def test_missing_budget_absent(self, synth_cache):
        enc = CLSEncoder(CLSSpace.from_yaml())
        recs = pq.build_records(synth_cache, enc, budgets=(290,),
                                slices=["Handwriting_00"])
        assert recs == []

    def test_slice_filter_excludes_other_dataset(self, synth_cache):
        enc = CLSEncoder(CLSSpace.from_yaml())
        recs = pq.build_records(synth_cache, enc, budgets=pq.BUDGETS,
                                slices=["Handwriting_00"])
        assert {pq.dataset_of(r.task_id) for r in recs} == {"Handwriting"}

    def test_windows_key_and_context(self, synth_cache):
        enc = CLSEncoder(CLSSpace.from_yaml())
        recs = pq.build_records(synth_cache, enc, budgets=pq.BUDGETS,
                                slices=["Handwriting_00", "Handwriting_01"])
        wk = pq.windows_key_of(recs)
        assert wk["Handwriting_00@60"] == "Handwriting_00"
        assert wk["Handwriting_01@120"] == "Handwriting_01"
        assert set(wk) == {r.task_id for r in recs}
        ctx = pq.context_of(recs, pq.BUDGETS)
        assert np.array_equal(ctx["Handwriting_00@60"], pq.ctx_vector(60))
        assert np.array_equal(ctx["Handwriting_01@120"], pq.ctx_vector(120))
        assert set(ctx) == set(wk)


# --------------------------------------------------------------- 消融指派
class TestAssignments:
    TIDS = [pq.engine_task_id(f"Handwriting_{i:02d}", b)
            for i in range(6) for b in pq.BUDGETS]

    def test_within_is_slice_derangement(self):
        amap = pq.within_assignment_slices(self.TIDS, 0)
        assert set(amap) == set(self.TIDS)
        perm = {}
        for t, donor in amap.items():
            s, b = pq.split_engine_task_id(t)
            ds, db = pq.split_engine_task_id(donor)
            assert db == b                      # 供体保留本对的预算
            assert ds != s                      # 无不动点
            perm.setdefault(s, ds)
            assert perm[s] == ds                # 同一切片排列跨预算一致
        assert sorted(perm.values()) == sorted(perm)     # 双射

    def test_within_requires_single_dataset(self):
        mixed = ["Handwriting_00@60", "Cricket_00@60"]
        with pytest.raises(AssertionError):
            pq.within_assignment_slices(mixed, 0)

    def test_within_seed_changes_permutation(self):
        perms = set()
        for s in range(5):
            amap = pq.within_assignment_slices(self.TIDS, s)
            perms.add(tuple(sorted(
                (pq.slice_of(k), pq.slice_of(v)) for k, v in amap.items())))
        assert len(perms) > 1

    def test_within_two_slices_is_seed_invariant(self):
        # S2 留出恰好 2 切片（04–05）：!2 = 1，唯一解是互换，seed 完全无作用
        tids = [pq.engine_task_id(f"Handwriting_{i:02d}", b)
                for i in (4, 5) for b in pq.BUDGETS]
        maps = [pq.within_assignment_slices(tids, s) for s in range(5)]
        assert all(m == maps[0] for m in maps)
        assert maps[0]["Handwriting_04@60"] == "Handwriting_05@60"
        assert maps[0]["Handwriting_05@290"] == "Handwriting_04@290"

    def test_within_strict_rejects_degenerate(self):
        tids = [pq.engine_task_id(f"Handwriting_{i:02d}", b)
                for i in (4, 5) for b in pq.BUDGETS]
        with pytest.raises(ValueError, match="seed"):
            pq.within_assignment_slices(tids, 0, strict=True)

    def test_within_strict_ok_from_three_slices(self):
        tids = [pq.engine_task_id(f"Handwriting_{i:02d}", 60) for i in range(3)]
        amap = pq.within_assignment_slices(tids, 0, strict=True)
        assert all(pq.slice_of(k) != pq.slice_of(v) for k, v in amap.items())

    def test_within_needs_two_slices(self):
        with pytest.raises(ValueError):
            pq.within_assignment_slices(["Handwriting_00@60"], 0)

    def test_n_derangements(self):
        # !0..!6 = 1, 0, 1, 2, 9, 44, 265（子阶乘；within/budget 档判读的可用排列数）
        assert [pq.n_derangements(n) for n in range(7)] == [1, 0, 1, 2, 9, 44, 265]

    def test_cross_donor_from_other_dataset(self):
        train = [pq.engine_task_id(f"{d}_{i:02d}", b)
                 for d in ("UniMiB-SHAR", "Cricket", "Handwriting")
                 for i in range(6) for b in pq.BUDGETS]
        amap = pq.cross_assignment_slices(self.TIDS, train, 0)
        assert set(amap) == set(self.TIDS)
        for t, donor in amap.items():
            _s, b = pq.split_engine_task_id(t)
            ds, db = pq.split_engine_task_id(donor)
            assert db == b
            assert pq.dataset_of(ds) in {"UniMiB-SHAR", "Cricket"}

    def test_cross_no_candidate_raises(self):
        train = [pq.engine_task_id(f"Handwriting_{i:02d}", 60) for i in range(6)]
        with pytest.raises(AssertionError):
            pq.cross_assignment_slices(["Handwriting_00@60"], train, 0)

    def test_budget_assignment_two_derangements(self):
        tids = [pq.engine_task_id("Handwriting_00", b) for b in pq.BUDGETS]
        a0, d0 = pq.budget_assignment(tids, 0)
        a1, d1 = pq.budget_assignment(tids, 1)
        assert d0 and d1
        # (60,120,290) → (120,290,60)
        assert a0 == {"Handwriting_00@60": "Handwriting_00@120",
                      "Handwriting_00@120": "Handwriting_00@290",
                      "Handwriting_00@290": "Handwriting_00@60"}
        # (60,120,290) → (290,60,120)
        assert a1 == {"Handwriting_00@60": "Handwriting_00@290",
                      "Handwriting_00@120": "Handwriting_00@60",
                      "Handwriting_00@290": "Handwriting_00@120"}

    def test_budget_assignment_keeps_slice(self):
        tids = [pq.engine_task_id(f"Handwriting_{i:02d}", b)
                for i in range(3) for b in pq.BUDGETS]
        amap, _ = pq.budget_assignment(tids, 0)
        for t, donor in amap.items():
            assert pq.slice_of(t) == pq.slice_of(donor)
            assert pq.split_engine_task_id(t)[1] != pq.split_engine_task_id(donor)[1]

    def test_budget_assignment_seed2_non_identity(self):
        tids = [pq.engine_task_id("Handwriting_00", b) for b in pq.BUDGETS]
        amap, is_der = pq.budget_assignment(tids, 2)
        assert any(k != v for k, v in amap.items())     # 非恒等
        moved = [k for k, v in amap.items() if k != v]
        assert is_der == (len(moved) == len(tids))

    def test_budget_assignment_deterministic(self):
        tids = [pq.engine_task_id("Handwriting_00", b) for b in pq.BUDGETS]
        assert pq.budget_assignment(tids, 7) == pq.budget_assignment(tids, 7)


# --------------------------------------------------------------- δ
class TestCalibrateDelta:
    def test_p95_branch(self):
        # n=21 → P95 索引 = int(round(0.95*20)) = 19 → 第 20 小 = 0.0713
        diffs = [0.001 * i for i in range(19)] + [0.0713, 0.09]
        assert pq.calibrate_delta(diffs, quantum=0.001) == pytest.approx(0.075)

    def test_quantum_floor_branch(self):
        # P95 = 0.011 < 2×0.0065 = 0.013 → 取量子下限 → 上取整 0.005 得 0.015
        diffs = [0.0005 * i for i in range(19)] + [0.011, 0.012]
        assert sorted(diffs)[19] == pytest.approx(0.011)
        assert pq.calibrate_delta(diffs, quantum=0.0065) == pytest.approx(0.015)

    def test_uses_absolute_values(self):
        pos = [0.001 * i for i in range(19)] + [0.0713, 0.09]
        neg = [-v for v in pos]
        assert pq.calibrate_delta(neg, quantum=0.001) == \
            pq.calibrate_delta(pos, quantum=0.001)

    def test_matches_2c_index_rule(self):
        rng = np.random.default_rng(0)
        diffs = list(rng.random(37) * 0.2)
        ds = sorted(abs(d) for d in diffs)
        p95 = ds[min(len(ds) - 1, int(round(0.95 * (len(ds) - 1))))]
        expect = float(np.ceil(max(p95, 0.0) / 0.005) * 0.005)
        assert pq.calibrate_delta(diffs, quantum=0.0) == pytest.approx(expect)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            pq.calibrate_delta([], quantum=0.001)


class TestDeltaMaps:
    def test_delta_map_for_fold(self):
        calib = {(d, b): 0.05 + 0.01 * i
                 for i, d in enumerate(("Handwriting", "Cricket"))
                 for b in pq.BUDGETS}
        slices = ["Handwriting_00", "Cricket_03"]
        dm = pq.delta_map_for_fold(calib, slices, pq.BUDGETS)
        assert set(dm) == {pq.engine_task_id(s, b) for s in slices for b in pq.BUDGETS}
        assert dm["Handwriting_00@60"] == pytest.approx(0.05)
        assert dm["Cricket_03@290"] == pytest.approx(0.06)

    def test_missing_dataset_budget_raises(self):
        calib = {("Handwriting", 60): 0.05}
        with pytest.raises(KeyError):
            pq.delta_map_for_fold(calib, ["Handwriting_00"], pq.BUDGETS)

    def test_load_delta_calibration(self, tmp_path):
        path = tmp_path / "delta_calibration.json"
        path.write_text(json.dumps({
            "Handwriting": {"60": {"delta": 0.075, "p95": 0.07, "quantum_floor": 0.006,
                                   "n_diffs": 30, "diff_median": 0.02, "delta_nd": 0.07},
                            "120": {"delta": 0.09}},
        }), encoding="utf-8")
        calib = pq.load_delta_calibration(path)
        assert calib[("Handwriting", 60)] == pytest.approx(0.075)
        assert calib[("Handwriting", 120)] == pytest.approx(0.09)
        assert all(isinstance(b, int) for _d, b in calib)


# --------------------------------------------------------------- 种子
class TestSeeds:
    def test_canonical_seeds(self):
        w, s = pq.canonical_seeds("Handwriting_00", "abc123")
        assert w == run_seed_for("Handwriting_00", "abc123")
        assert s == run_seed_for("Handwriting_00", "train_eval_split")

    def test_split_seed_shared_across_cls(self):
        assert pq.canonical_seeds("Handwriting_00", "a")[1] == \
            pq.canonical_seeds("Handwriting_00", "b")[1]

    def test_noise_seed_convention(self):
        # 2c 噪声种子约定：f"{cls_key}:noise{rep}"
        assert pq.noise_seed("Handwriting_00", "abc", 1) == \
            run_seed_for("Handwriting_00", "abc:noise1")
        assert pq.noise_seed("Handwriting_00", "abc", 1) != \
            pq.noise_seed("Handwriting_00", "abc", 2)
        assert pq.noise_seed("Handwriting_00", "abc", 1) != \
            pq.canonical_seeds("Handwriting_00", "abc")[0]


# --------------------------------------------------------------- CLS 目录
class TestCatalog:
    def test_shape_and_origins(self):
        space = CLSSpace.from_yaml()
        cat = pq.catalog_for_slice(space, "Handwriting_00")
        assert len(cat) == 20
        origins = [o for o, _c, _k in cat]
        assert origins.count("shared") == 10
        assert origins.count("random") == 10
        assert origins[:10] == ["shared"] * 10

    def test_shared_from_catalog_file(self):
        space = CLSSpace.from_yaml()
        cat = pq.catalog_for_slice(space, "Handwriting_00")
        want = [k for _c, k in load_probe_cls(pq.SHARED_CATALOG, 10)[0]]
        assert [k for o, _c, k in cat if o == "shared"] == want

    def test_random_unique_and_disjoint(self):
        space = CLSSpace.from_yaml()
        cat = pq.catalog_for_slice(space, "Handwriting_00")
        shared = {k for o, _c, k in cat if o == "shared"}
        rnd = [k for o, _c, k in cat if o == "random"]
        assert len(set(rnd)) == 10
        assert set(rnd) & shared == set()
        assert all(cls_key(c) == k for _o, c, k in cat)

    def test_deterministic_and_slice_specific(self):
        space = CLSSpace.from_yaml()
        a = pq.catalog_for_slice(space, "Handwriting_00")
        b = pq.catalog_for_slice(space, "Handwriting_00")
        c = pq.catalog_for_slice(space, "Handwriting_01")
        assert [k for _o, _c, k in a] == [k for _o, _c, k in b]
        assert [k for o, _c, k in a if o == "random"] != \
            [k for o, _c, k in c if o == "random"]

    def test_ggs_is_in_space(self):
        space = CLSSpace.from_yaml()
        assert space.validate(pq.ggs_cls())


# --------------------------------------------------------------- 配置
class TestConfig:
    def test_pq_section(self):
        cfg = pq.load_pq_config()
        assert cfg["budgets"] == [60, 120, 290]
        assert cfg["n_slices_per_dataset"] == 6
        assert cfg["datasets"] == list(pq.PQ_DATASETS)
        assert cfg["training_only"] == list(pq.TRAINING_ONLY)
        assert cfg["settf_t_max"] == 152
        assert cfg["settf_windows"] == 100
        assert cfg["ctx_encoding"] == "onehot"
        assert cfg["noise_probe"] == {"slices": 2, "reps": [1, 2]}
        # 未标定时是空表；`32_pq_generate_rprime.py --calibrate --apply-delta` 回填后
        # 是 8 数据集 × 3 预算的全表（覆盖度闸保证不会是半张）。两种都合法，
        # 别让"跑过标定"这件事把本用例判红。
        dpb = cfg["delta_per_dataset_budget"]
        assert dpb == {} or (
            set(dpb) == set(pq.PQ_DATASETS)
            and all(set(v) == set(pq.BUDGETS)
                    and all(isinstance(x, float) for x in v.values())
                    for v in dpb.values())), dpb
        assert cfg["evolution"] == {"ks": 2000, "kp": 50, "generations": 20,
                                    "p_cross": 0.8, "p_mut": 0.2, "top_k": 3}

    def test_train_yaml_budgets_consistent(self):
        import yaml
        train = yaml.safe_load(
            (pq.ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
        assert train["budgets"] == list(pq.BUDGETS)
        assert train["early_iters"] == 290      # 既有口径不动（铁律 2）
        assert train["full_iters"] == 2320

    def test_inconsistent_budgets_fail_fast(self, tmp_path):
        import yaml
        labels = yaml.safe_load(
            (pq.ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
        labels["pq"]["budgets"] = [60, 120]
        lp = tmp_path / "labels.yaml"
        lp.write_text(yaml.safe_dump(labels), encoding="utf-8")
        with pytest.raises(AssertionError):
            pq.load_pq_config(labels_path=lp)


# --------------------------------------------------------------- 基线带
class TestBaselineBands:
    def test_handwriting_bands(self):
        bands = pq.baseline_bands("Handwriting")
        for s in pq.pq_slice_ids("Handwriting"):
            mean, band = bands[s]
            assert 0.0 <= mean <= 1.0
            assert band > 0.0

    def test_scp2_has_three_slices(self):
        assert pq.pq_slice_ids("SelfRegulationSCP2") == [
            "SelfRegulationSCP2_00", "SelfRegulationSCP2_01", "SelfRegulationSCP2_02"]

    def test_all_pq_datasets_covered(self):
        for d in pq.PQ_DATASETS:
            bands = pq.baseline_bands(d)
            assert set(pq.pq_slice_ids(d)) <= set(bands)

    def test_missing_slice_raises(self):
        # 拿错缓存（候选池基线里没有 Handwriting 切片）→ fail-fast
        with pytest.raises(ValueError):
            pq.baseline_bands("Handwriting", path=pq.BASELINE_CANDIDATE)

    def test_band_formula(self):
        import statistics
        rows = [r for r in pq.read_jsonl(pq.BASELINE_UEA)
                if r["task_id"] == "Handwriting_00" and r.get("r_prime") is not None]
        vals = [float(r["r_prime"]) for r in rows]
        std = statistics.stdev(vals)
        quantum = max(1.0 / r["n_eval"] for r in rows if r.get("n_eval"))
        mean, band = pq.baseline_bands("Handwriting")["Handwriting_00"]
        assert mean == pytest.approx(statistics.fmean(vals))
        assert band == pytest.approx(2.0 * max(std, quantum))


# --------------------------------------------------------------- tasks.json
class TestTasksJson:
    def _dataset(self):
        rng = np.random.default_rng(0)
        y = np.array([0, 1, 2] * 40)
        X = rng.random((len(y), 12, 3)).astype(np.float32)
        idx = tuple(int(v) for v in np.where(y < 2)[0])
        slices = [SlicedTask(task_id=f"Cricket_{i:02d}", pool="Cricket",
                             classes=(0, 1), channels=(0, 2),
                             sample_indices=idx, tag=f"t{i}")
                  for i in range(2)]
        return pq.PQDataset(name="Cricket", X=X, y=y, entry=None, registry=None,
                            slices=slices)

    def test_round_trip(self, tmp_path):
        path = tmp_path / "tasks.json"
        pq.write_tasks_json({"Cricket": self._dataset()}, path)
        loaded = pq.load_tasks_json(path)
        assert set(loaded) == {"Cricket_00", "Cricket_01"}
        rec = loaded["Cricket_00"]
        assert rec["task_id"] == "Cricket_00"
        assert rec["classes"] == [0, 1]
        assert rec["channels"] == [0, 2]
        assert rec["dataset"] == "Cricket"
        assert rec["task_type"] == "classification"
        assert rec["n_train"] + rec["n_eval"] == len(rec["sample_indices"])
        assert rec["n_eval"] > 0 and rec["n_train"] > 0


# --------------------------------------------------------------- 噪声模型
def _prob_rows(budgets=(60, 120), n_slices=2, n_cls=5, reps=(1, 2)):
    """合成 (dataset, budget) 噪声探针材料：k=3（canonical + 2 rep）。"""
    rng = np.random.default_rng(1)
    cache_rows, noise_rows = [], []
    for si in range(n_slices):
        tid = f"Handwriting_{si:02d}"
        for b in budgets:
            for ci in range(n_cls):
                key = f"cls{ci:02d}"
                base = 0.5 + 0.01 * ci + 0.001 * b
                cache_rows.append({"dataset": "Handwriting", "task_id": tid,
                                   "cls_key": key, "iters": b, "seed": 0,
                                   "r_prime": base + float(rng.normal(0, 0.02)),
                                   "n_eval": 300 + 200 * si})
                for rep in reps:
                    noise_rows.append({"dataset": "Handwriting", "task_id": tid,
                                       "cls_key": key, "iters": b, "seed": rep,
                                       "r_prime": base + float(rng.normal(0, 0.02)),
                                       "n_eval": 300 + 200 * si})
    strict = {(pq.engine_task_id(r["task_id"], r["iters"]), r["cls_key"]): False
              for r in cache_rows}
    return cache_rows, noise_rows, strict


class TestProbModelPQ:
    def test_per_dataset_budget_models(self):
        rows = _prob_rows()
        pm = pq.ProbModelPQ(*rows, (60, 120), "student_t")
        assert set(pm.models) == {("Handwriting", 60), ("Handwriting", 120)}

    def test_probability_monotone_and_symmetric(self):
        pm = pq.ProbModelPQ(*_prob_rows(), (60, 120), "student_t")
        tid = "Handwriting_00@60"
        p0 = pm(tid, "cls00", "cls01", 0.0)
        p1 = pm(tid, "cls00", "cls01", 0.05)
        p2 = pm(tid, "cls00", "cls01", 0.20)
        assert p0 == pytest.approx(0.5, abs=1e-9)
        assert 0.5 < p1 < p2 <= 1.0
        assert pm(tid, "cls00", "cls01", -0.05) == pytest.approx(1.0 - p1, abs=1e-9)

    def test_empirical_kind(self):
        pm = pq.ProbModelPQ(*_prob_rows(), (60, 120), "empirical")
        assert pm.kind == "empirical"
        assert 0.0 <= pm("Handwriting_00@60", "cls00", "cls01", 0.05) <= 1.0

    def test_task_scale_factor_applied(self):
        pm = pq.ProbModelPQ(*_prob_rows(), (60, 120), "student_t")
        # n_eval 小的任务尺度更大（计数成分 ∝ 1/√n_eval，κ=0.595）
        assert pm.factor["Handwriting_00@60"] > pm.factor["Handwriting_01@60"]

    def test_to_json_serializable(self):
        pm = pq.ProbModelPQ(*_prob_rows(), (60, 120), "student_t")
        json.dumps(pm.to_json())
        assert pm.to_json()["kappa"] == pytest.approx(pq.KAPPA)

    def test_missing_strict_entry_fails_fast(self):
        cache_rows, noise_rows, strict = _prob_rows()
        strict.pop(("Handwriting_00@60", "cls00"))
        with pytest.raises(KeyError):
            pq.ProbModelPQ(cache_rows, noise_rows, strict, (60, 120), "student_t")


# --------------------------------------------------------------- 切片 id 归一化
def _write_tasks(tmp_path, name="Cricket", n=6):
    """写一份 n 切片的 tasks.json，返回 load_tasks_json 的输出（{slice_id: task_dict}）。"""
    rng = np.random.default_rng(0)
    y = np.array([0, 1, 2] * 40)
    X = rng.random((len(y), 12, 3)).astype(np.float32)
    idx = tuple(int(v) for v in np.where(y < 2)[0])
    slices = [SlicedTask(task_id=f"{name}_{i:02d}", pool=name, classes=(0, 1),
                         channels=(0, 2), sample_indices=idx, tag=f"t{i}")
              for i in range(n)]
    ds = pq.PQDataset(name=name, X=X, y=y, entry=None, registry=None, slices=slices)
    path = tmp_path / "tasks.json"
    pq.write_tasks_json({name: ds}, path)
    return pq.load_tasks_json(path)


class TestAsSliceIds:
    def test_tasks_json_shape(self, tmp_path):
        """回归：{slice_id: task_dict} 曾被递归进 classes:[0,1] → TypeError。"""
        tasks = _write_tasks(tmp_path)
        assert pq._as_slice_ids(tasks) == [f"Cricket_{i:02d}" for i in range(6)]

    def test_single_task_dict(self, tmp_path):
        tasks = _write_tasks(tmp_path, n=1)
        assert pq._as_slice_ids(tasks["Cricket_00"]) == ["Cricket_00"]

    def test_s2_split_on_tasks_json(self, tmp_path):
        train, test = pq.s2_split(_write_tasks(tmp_path))
        assert train == [f"Cricket_{i:02d}" for i in range(4)]
        assert test == ["Cricket_04", "Cricket_05"]

    def test_build_records_on_tasks_json(self, tmp_path, synth_cache):
        tasks = _write_tasks(tmp_path, name="Handwriting", n=2)
        enc = CLSEncoder(CLSSpace.from_yaml())
        recs = pq.build_records(synth_cache, enc, budgets=pq.BUDGETS, slices=tasks)
        assert {r.task_id for r in recs} == {
            f"Handwriting_{i:02d}@{b}" for i in range(2) for b in (60, 120)}

    def test_delta_map_on_tasks_json(self, tmp_path):
        tasks = _write_tasks(tmp_path, n=6)
        calib = {("Cricket", b): 0.08 for b in pq.BUDGETS}
        dm = pq.delta_map_for_fold(calib, tasks, pq.BUDGETS)
        assert len(dm) == 18
        assert dm["Cricket_05@290"] == pytest.approx(0.08)

    def test_other_shapes_still_work(self):
        assert pq._as_slice_ids("Handwriting_03@120") == ["Handwriting_03"]
        assert pq._as_slice_ids({"Cricket": ["Cricket_00", "Cricket_01"]}) == [
            "Cricket_00", "Cricket_01"]
        t = SlicedTask(task_id="Cricket_00", pool="Cricket", classes=(0, 1),
                       channels=(0,), sample_indices=(0, 1))
        assert pq._as_slice_ids([t, "Cricket_01"]) == ["Cricket_00", "Cricket_01"]
        assert pq._as_slice_ids(dataclasses.asdict(t)) == ["Cricket_00"]

    def test_bad_leaf_raises_value_error(self):
        with pytest.raises(ValueError, match="无法解释成切片 id"):
            pq._as_slice_ids([1, 2])
        with pytest.raises(ValueError, match="无法解释成切片 id"):
            pq._as_slice_ids({"Cricket": 3})
        with pytest.raises(ValueError, match="无法解释成切片 id"):
            pq.s2_split([None])
