"""MX-3-PQ 阶段 33（scripts/33_pq_comparator_loo.py）单测：折特征守卫 / 臂工厂 / 消融块 / 判读。

协议：results/mx3_pq/PROTOCOL.md §3–§6；接口契约见
.superpowers/sdd/autocts-e-autocts-autocts-e-autocts-refactored-moler/interfaces.md
「Task 6（script 33）」「Task 6 → Task 7 saved-model format」。

本文件不加载任何真实数据集（aeon / UEA 缓存不参与）：折特征段用微型未训练 TS2Vec
在合成 X 上跑，比较器段用玩具引擎（2 数据集 × 3 切片 × 2 预算），判读段用手写 run
JSON。冒烟段跑 run_smoke 的最小规模（2 数据集 × 3 切片 × 4 CLS × 1 epoch）。
"""
import importlib
import json

import numpy as np
import pytest
import torch

from autocls.comparator.tclsc_engine import ClsRecord, TCLSCEngine, with_reversed
from autocls.comparator.tclsc_model import TCLSC
from autocls.task_embed import TS2VecWrapper, TaskSetEncoder
from scripts import pq_common as pq

m33 = importlib.import_module("scripts.33_pq_comparator_loo")

TASK_TYPE = "classification"


# ================================================================ 折特征（Stage A）
class TestFoldFitGuard:
    """铁律 6：折 TS2Vec 的 fit 池绝不含考侧数据集。"""

    def test_target_in_fit_pool_raises(self):
        with pytest.raises(AssertionError):
            m33.assert_fold_fit_pool("Handwriting", ["Handwriting", "Cricket"])

    def test_clean_pool_ok(self):
        m33.assert_fold_fit_pool("Handwriting", ["Cricket", "UniMiB-SHAR"])

    def test_all8_fold_is_exempt(self):
        # S2 上界臂的 all8 折**故意**含全部 8 个数据集（非 zero-shot，单独报）
        m33.assert_fold_fit_pool(m33.ALL8_FOLD, list(pq.PQ_DATASETS))


@pytest.fixture
def tiny_ts2vec():
    """微型未训练单变量 TS2Vec（frozen）——allow_unfit 路径，秒级。"""
    w = TS2VecWrapper(input_dims=1, output_dims=8, hidden_dims=4, depth=2, device="cpu")
    w.freeze()
    return w


class TestSliceFeature:
    """task_features/<slice>.npz 的键 / dtype / 形状（interfaces「Task 6 → Task 7」）。"""

    X = np.random.default_rng(0).standard_normal((5, 40, 2)).astype(np.float32)

    def _feat(self, tiny_ts2vec, t_max):
        enc = TaskSetEncoder(dim_input=8, nz=8)
        return m33.encode_slice_feature(
            self.X, TASK_TYPE, tiny_ts2vec, enc, sample_num=3, seed=0,
            t_max=t_max, n_classes=4, allow_unfit=True)

    def test_keys_dtypes_shapes(self, tiny_ts2vec):
        feat = self._feat(tiny_ts2vec, 10)
        assert set(feat) == {"windows_repr", "E_init", "META"}
        assert feat["windows_repr"].dtype == np.float16
        assert feat["windows_repr"].shape == (3, 10, 8)      # pool_time_axis 生效
        assert feat["E_init"].dtype == np.float32
        assert feat["E_init"].shape == (8,)
        assert feat["META"].dtype == np.float32
        assert feat["META"].shape == (4,)

    def test_pool_time_axis_identity_when_short(self, tiny_ts2vec):
        feat = self._feat(tiny_ts2vec, 152)
        assert feat["windows_repr"].shape == (3, 40, 8)      # T=40 ≤ 152，不池化

    def test_meta_vector_semantics(self, tiny_ts2vec):
        feat = self._feat(tiny_ts2vec, 10)
        meta = feat["META"]
        np.testing.assert_allclose(meta[0], np.log10(40), rtol=1e-5)   # log10 T
        np.testing.assert_allclose(meta[1], np.log10(2), rtol=1e-5)    # log10 C
        assert meta[2] == 4.0                                          # n_classes
        np.testing.assert_allclose(meta[3], np.log10(5), rtol=1e-5)    # log10 N


# ================================================================ 臂工厂（Stage B）
EXPECTED_ARMS = {
    "main": ("E+ctx", 3), "autocts_style": ("E+ctx", 3), "wo_ts2vec": ("E+ctx", 3),
    "wo_settf": ("E+ctx", 3), "wo_shared": ("E+ctx", 3), "cls_only": ("none", 0),
    "cls_b": ("ctx", 3), "soft": ("E+ctx", 3), "onehot_oracle": ("ctx", 11),
    "main_s2": ("E+ctx", 3),
}
CFG = {"embedding_dim": 16, "fz": 16, "lr": 1e-3, "batch_size": 8,
       "epochs": 2, "curriculum_epochs": 1}


class TestArmFactory:
    def test_arm_names_exact(self):
        assert set(m33.ARMS) == set(EXPECTED_ARMS)

    @pytest.mark.parametrize("name", sorted(EXPECTED_ARMS))
    def test_branch_and_ctx_dim(self, name):
        spec = m33.ARMS[name]
        cdim = m33.ctx_dim_for(spec, pq.BUDGETS, pq.PQ_DATASETS)
        assert (spec.task_branch, cdim) == EXPECTED_ARMS[name]
        model = m33.build_model(spec, cls_dim=5, ctx_dim=cdim, nz=8, dim_input=8,
                                cfg=CFG, seed=0, init_path=None)
        assert isinstance(model, TCLSC)
        assert model.task_branch == spec.task_branch
        assert model.ctx_dim == cdim
        assert (model.set_encoder is not None) == (spec.encoder is not None)

    def test_wo_settf_uses_mean_pool(self):
        spec = m33.ARMS["wo_settf"]
        model = m33.build_model(spec, cls_dim=5, ctx_dim=3, nz=8, dim_input=8,
                                cfg=CFG, seed=0, init_path=None)
        assert type(model.set_encoder).__name__ == "MeanPoolTaskEncoder"

    def test_s2_arms_marked(self):
        assert m33.ARMS["onehot_oracle"].split == "S2"
        assert m33.ARMS["main_s2"].split == "S2"
        assert m33.ARMS["main"].split == "S1"

    def test_e_source_stays_in_binding_enum(self):
        # interfaces 绑定枚举：("fold"|"per_dataset"|"unfit"|null)——"fold_all8" 不合法
        assert set(m33.E_SOURCES) == {"fold", "per_dataset", "unfit", None}
        for spec in m33.ARMS.values():
            assert spec.e_source in m33.E_SOURCES, spec.name

    def test_feature_fold_resolution(self):
        # S1 折内 E → 折靶名；main_s2 → all8（fold 是 "S2"，光看它找不到目录）
        assert m33.feature_fold_of(m33.ARMS["main"], "Cricket") == "Cricket"
        assert m33.feature_fold_of(m33.ARMS["main_s2"], m33.S2_FOLD) == m33.ALL8_FOLD
        assert m33.ARMS["main_s2"].e_source == "fold"
        # per_dataset / unfit / 无 E 的臂：按契约写 null
        for arm in ("autocts_style", "wo_ts2vec", "cls_b", "cls_only"):
            assert m33.feature_fold_of(m33.ARMS[arm], "Cricket") is None

    def test_iron_law_6_flags(self):
        assert m33.ARMS["main"].iron_law_6_compliant is True
        assert m33.ARMS["autocts_style"].iron_law_6_compliant is False
        for arm in ("main_s2", "onehot_oracle"):
            assert m33.ARMS[arm].iron_law_6_compliant == "s2_oracle"

    def test_default_seeds_and_folds(self):
        assert m33.ARMS["main"].seeds == (0, 1, 2)
        assert all(m33.ARMS[a].seeds == (0,) for a in EXPECTED_ARMS if a != "main")
        trio = {"Handwriting", "UniMiB-SHAR", "Cricket"}
        for arm in ("wo_ts2vec", "wo_shared", "soft"):
            assert set(m33.default_folds(m33.ARMS[arm])) == trio
        assert set(m33.default_folds(m33.ARMS["main"])) == set(pq.FOLD_TARGETS)
        assert m33.default_folds(m33.ARMS["onehot_oracle"]) == (m33.S2_FOLD,)

    def test_wo_shared_filters_random_origin(self):
        recs = [ClsRecord("Cricket_00@60", "k1", "shared", np.zeros(3), 0.5),
                ClsRecord("Cricket_00@60", "k2", "random", np.zeros(3), 0.6)]
        kept = m33.filter_train_records(m33.ARMS["wo_shared"], recs)
        assert [r.key for r in kept] == ["k2"]
        assert len(m33.filter_train_records(m33.ARMS["main"], recs)) == 2

    def test_wo_shared_disables_curriculum(self):
        assert m33.curriculum_for(m33.ARMS["wo_shared"], 40) == 0
        assert m33.curriculum_for(m33.ARMS["main"], 40) == 40


class TestContextBuilders:
    def test_budget_onehot_for_main(self):
        tid = pq.engine_task_id("Cricket_00", 290)
        ctx = m33.context_for(m33.ARMS["main"], [tid], pq.BUDGETS, pq.PQ_DATASETS)
        np.testing.assert_array_equal(ctx[tid], pq.ctx_vector(290))

    def test_onehot_oracle_is_dataset_plus_budget(self):
        tid = pq.engine_task_id("Cricket_00", 120)
        ctx = m33.context_for(m33.ARMS["onehot_oracle"], [tid], pq.BUDGETS,
                              pq.PQ_DATASETS)
        v = ctx[tid]
        assert v.shape == (11,) and v.dtype == np.float32
        assert v.sum() == 2.0
        assert v[pq.PQ_DATASETS.index("Cricket")] == 1.0
        assert v[len(pq.PQ_DATASETS) + pq.BUDGETS.index(120)] == 1.0

    def test_cls_only_has_no_context(self):
        assert m33.context_for(m33.ARMS["cls_only"], ["Cricket_00@60"],
                               pq.BUDGETS, pq.PQ_DATASETS) is None


# ================================================================ 消融块
BUDGETS2 = (60, 120)
TOY_DS = ("Handwriting", "Cricket")


def _toy_records(rng, n_slices=3):
    recs = []
    for d in TOY_DS:
        for i in range(n_slices):
            for b in BUDGETS2:
                for j in range(4):
                    recs.append(ClsRecord(
                        task_id=pq.engine_task_id(f"{d}_{i:02d}", b), key=f"k{j}",
                        origin="shared" if j < 2 else "random",
                        vec=rng.standard_normal(5).astype(np.float32),
                        r=0.50 + 0.05 * j + 0.01 * i + 0.002 * b / 60))
    return recs


def _toy_engine(task_branch, *, n_slices=3, seed=0):
    """玩具引擎 + 记录（合成窗口，CPU）。task_branch 决定 E/ctx 支路是否存在。"""
    rng = np.random.default_rng(seed)
    recs = _toy_records(rng, n_slices)
    tids = sorted({r.task_id for r in recs})
    windows = {pq.slice_of(t): rng.standard_normal((2, 6, 8)).astype(np.float32)
               for t in tids}
    ctx = (None if task_branch == "none" else
           {t: pq.ctx_vector(pq.split_engine_task_id(t)[1], BUDGETS2) for t in tids})
    torch.manual_seed(0)
    set_enc = TaskSetEncoder(dim_input=8, nz=8) if task_branch == "E+ctx" else None
    model = TCLSC(cls_dim=5, set_encoder=set_enc, embedding_dim=8, fz=8,
                  ctx_dim=0 if ctx is None else len(BUDGETS2), task_branch=task_branch)
    engine = TCLSCEngine(model, windows if task_branch == "E+ctx" else {},
                         device="cpu", batch_size=32,
                         windows_key_of=pq.windows_key_of(recs), context_of=ctx)
    return engine, recs, tids


def _toy_eval_set(recs, tids):
    test_tids = [t for t in tids if pq.dataset_of(t) == "Handwriting"]
    train_tids = [t for t in tids if pq.dataset_of(t) == "Cricket"]
    test_recs = [r for r in recs if r.task_id in set(test_tids)]
    fwd = m33.forward_pairs_pq(test_recs, TASK_TYPE, {t: 0.0 for t in test_tids})
    return with_reversed(fwd), len(fwd), test_tids, train_tids


AGG_KEYS = {"rows", "n_rows", "mean_delta", "min_delta", "max_delta", "n_sig",
            "pass", "tier"}


class TestAblationBlock:
    def test_full_block_on_e_arm(self):
        engine, recs, tids = _toy_engine("E+ctx")
        directed, n_fwd, test_tids, train_tids = _toy_eval_set(recs, tids)
        blocks, correct = m33.ablation_block(
            engine, directed, n_fwd, test_tids, train_tids, BUDGETS2)
        assert set(blocks) == {"normal", "within", "cross", "budget", "zero_E",
                               "zero_ctx"}
        for key in ("within", "cross", "budget", "zero_E", "zero_ctx"):
            assert AGG_KEYS <= set(blocks[key]), key
            assert blocks[key]["tier"] in ("pass", "detectable", "zero", "ambiguous")
        assert blocks["within"]["n_rows"] == 5
        assert blocks["cross"]["n_rows"] == 5
        assert blocks["budget"]["n_rows"] == 5
        assert all("is_derangement" in r for r in blocks["budget"]["rows"])
        # zero 档没有随机性：单行 + 显式标注（绝不报 5 行假方差）
        assert blocks["zero_E"]["n_rows"] == 1
        assert blocks["zero_E"]["deterministic"] is True
        assert blocks["zero_ctx"]["n_rows"] == 1
        assert blocks["normal"]["n"] == len(directed)
        assert set(blocks["normal"]["by_budget"]) == {str(b) for b in BUDGETS2}
        assert blocks["normal"]["by_kind"]
        assert correct["normal"].dtype == np.bool_
        assert len(correct["normal"]) == n_fwd
        assert "cross_s0" in correct and "budget_s0" in correct

    def test_cross_channel_is_e_by_default(self):
        engine, recs, tids = _toy_engine("E+ctx")
        directed, n_fwd, test_tids, train_tids = _toy_eval_set(recs, tids)
        blocks, _ = m33.ablation_block(engine, directed, n_fwd, test_tids,
                                       train_tids, BUDGETS2)
        assert blocks["cross"]["channel"] == "E"

    def test_arm_without_e_gives_null_e_ablations(self):
        engine, recs, tids = _toy_engine("ctx")
        directed, n_fwd, test_tids, train_tids = _toy_eval_set(recs, tids)
        blocks, _ = m33.ablation_block(engine, directed, n_fwd, test_tids,
                                       train_tids, BUDGETS2)
        assert blocks["within"] is None
        assert blocks["zero_E"] is None
        assert blocks["cross"] is None          # ctx 只编码预算：跨数据集换 ctx = 恒等
        assert blocks["budget"] is not None
        assert blocks["zero_ctx"] is not None

    def test_ctx_cross_channel_for_dataset_identity_ctx(self):
        engine, recs, tids = _toy_engine("ctx")
        directed, n_fwd, test_tids, train_tids = _toy_eval_set(recs, tids)
        blocks, _ = m33.ablation_block(engine, directed, n_fwd, test_tids,
                                       train_tids, BUDGETS2, cross_channel="ctx")
        assert blocks["cross"] is not None
        assert blocks["cross"]["channel"] == "ctx"
        assert blocks["cross"]["n_rows"] == 5

    def test_task_blind_arm_gives_all_null(self):
        engine, recs, tids = _toy_engine("none")
        directed, n_fwd, test_tids, train_tids = _toy_eval_set(recs, tids)
        blocks, _ = m33.ablation_block(engine, directed, n_fwd, test_tids,
                                       train_tids, BUDGETS2)
        for key in ("within", "cross", "budget", "zero_E", "zero_ctx"):
            assert blocks[key] is None, key
        assert blocks["normal"]["acc"] >= 0.0

    def test_two_slice_within_is_single_degenerate_row(self):
        """!2 = 1：2 切片时错位排列唯一，seed 无作用——只报单行并标 degenerate。"""
        engine, recs, tids = _toy_engine("E+ctx", n_slices=2)
        directed, n_fwd, test_tids, train_tids = _toy_eval_set(recs, tids)
        blocks, _ = m33.ablation_block(engine, directed, n_fwd, test_tids,
                                       train_tids, BUDGETS2)
        assert blocks["within"]["n_rows"] == 1
        assert blocks["within"]["degenerate"] is True
        assert blocks["cross"]["n_rows"] == 5      # cross 不受切片数退化影响

    def test_single_budget_skips_budget_ablation(self):
        engine, recs, tids = _toy_engine("E+ctx")
        directed, n_fwd, test_tids, train_tids = _toy_eval_set(recs, tids)
        blocks, _ = m33.ablation_block(engine, directed, n_fwd, test_tids,
                                       train_tids, (60,))
        assert blocks["budget"] is None


# ================================================================ 判读（Stage C）
class TestWindowsCacheEviction:
    """特征目录一换即驱逐：45 切片 × (100,152,256) float32 ≈ 0.65 GiB/目录。"""

    def test_switching_feature_dir_evicts(self, tmp_path):
        m33.run_smoke(tmp_path, epochs=1, arms=("cls_only",),
                      datasets=("Handwriting", "Cricket"), n_slices=3, n_cls=4,
                      device="cpu", settf_windows=2, t_max=6, folds=("Handwriting",))
        ctx = m33.PQContext(tmp_path, device="cpu", epochs=1, settf_windows=2,
                            smoke=True)
        slices = ["Handwriting_00", "Handwriting_01"]
        ctx.windows(m33.ARMS["main"], "Handwriting", slices)          # folds/…/task_features
        assert set(ctx._windows_cache) == set(slices)
        first_dir = ctx._windows_dir
        ctx.windows(m33.ARMS["wo_ts2vec"], "Handwriting", ["Handwriting_00"])  # …_unfit
        assert ctx._windows_dir != first_dir
        assert set(ctx._windows_cache) == {"Handwriting_00"}          # 旧目录已清空

    def test_same_dir_keeps_cache(self, tmp_path):
        m33.run_smoke(tmp_path, epochs=1, arms=("cls_only",),
                      datasets=("Handwriting", "Cricket"), n_slices=3, n_cls=4,
                      device="cpu", settf_windows=2, t_max=6, folds=("Handwriting",))
        ctx = m33.PQContext(tmp_path, device="cpu", epochs=1, settf_windows=2,
                            smoke=True)
        ctx.windows(m33.ARMS["main"], "Handwriting", ["Handwriting_00"])
        ctx.windows(m33.ARMS["main"], "Handwriting", ["Handwriting_01"])
        assert set(ctx._windows_cache) == {"Handwriting_00", "Handwriting_01"}


class TestWindowsCalibration:
    """预注册旋钮的外推：只有训练部分乘 epoch 数，一次性开销（消融/配对）加一次。"""

    def test_only_training_scales_with_epochs(self):
        # 整跑 100 s = 训练 1 epoch 2 s + 开销 98 s；200 epoch → (2×200 + 98)/60
        assert m33.project_run_minutes(2.0, 100.0, 200) == pytest.approx(498.0 / 60.0)

    def test_no_overhead_reduces_to_plain_scaling(self):
        assert m33.project_run_minutes(3.0, 3.0, 200) == pytest.approx(600.0 / 60.0)

    def test_naive_scaling_would_overstate(self):
        naive = 100.0 * 200 / 60.0
        assert m33.project_run_minutes(2.0, 100.0, 200) < naive


class TestCurriculumClamp:
    def test_epochs_override_clamps_curriculum(self, tmp_path):
        m33.run_smoke(tmp_path, epochs=1, arms=("cls_only",),
                      datasets=("Handwriting", "Cricket"), n_slices=3, n_cls=4,
                      device="cpu", settf_windows=2, t_max=6, folds=("Handwriting",))
        ctx = m33.PQContext(tmp_path, device="cpu", epochs=2, settf_windows=2,
                            smoke=True)
        assert ctx.cfg["epochs"] == 2
        # min(40, max(1, 2 // 2)) = 1：留出后半程给 random，否则薄目录下 0 对可配
        assert ctx.cfg["curriculum_epochs"] == 1
        assert m33.PQContext(tmp_path, device="cpu", epochs=1, smoke=True,
                             settf_windows=2).cfg["curriculum_epochs"] == 1
        assert m33.PQContext(tmp_path, device="cpu", epochs=10, smoke=True,
                             settf_windows=2).cfg["curriculum_epochs"] == 5
        assert m33.PQContext(tmp_path, device="cpu", epochs=200, smoke=True,
                             settf_windows=2).cfg["curriculum_epochs"] == 40
        full = m33.PQContext(tmp_path, device="cpu", settf_windows=2, smoke=True)
        assert full.cfg["curriculum_epochs"] == 40    # 不覆盖时原样


class TestVerdictCode:
    BASE = dict(budget_tiers=["pass"] * 7, pooled_p=0.5, q3_delta=0.0, q3_p=0.5,
                a8_cross_tier=None)

    def test_e_alive(self):
        v = m33.verdict_code(**{**self.BASE, "cross_tiers": ["pass"] * 5 + ["zero"] * 2,
                                "pooled_p": 0.01, "q3_delta": 0.02, "q3_p": 0.01})
        assert v["code"] == "E-ALIVE"
        assert v["conditions"]["e_alive"] is True

    def test_e_alive_needs_q3(self):
        v = m33.verdict_code(**{**self.BASE, "cross_tiers": ["pass"] * 5 + ["zero"] * 2,
                                "pooled_p": 0.01, "q3_delta": 0.02, "q3_p": 0.40})
        assert v["code"] != "E-ALIVE"

    def test_e_detect(self):
        v = m33.verdict_code(**{**self.BASE,
                                "cross_tiers": ["detectable"] * 4 + ["zero"] * 3,
                                "pooled_p": 0.20})
        assert v["code"] == "E-DETECT"

    def test_label_blind(self):
        v = m33.verdict_code(**{**self.BASE, "cross_tiers": ["zero"] * 7,
                                "a8_cross_tier": "zero"})
        assert v["code"] == "LABEL-BLIND"

    def test_e_pathway(self):
        v = m33.verdict_code(**{**self.BASE, "cross_tiers": ["zero"] * 7,
                                "a8_cross_tier": "pass"})
        assert v["code"] == "E-PATHWAY"

    def test_b_dead(self):
        v = m33.verdict_code(**{**self.BASE, "cross_tiers": ["ambiguous"] * 7,
                                "budget_tiers": ["zero"] * 7})
        assert v["code"] == "B-DEAD"

    def test_ambiguous_lists_unmet(self):
        v = m33.verdict_code(**{**self.BASE, "cross_tiers": ["ambiguous"] * 7})
        assert v["code"] == "AMBIGUOUS"
        assert v["unmet"]

    def test_b_dead_flag_is_reported_even_when_code_differs(self):
        v = m33.verdict_code(**{**self.BASE, "cross_tiers": ["zero"] * 7,
                                "budget_tiers": ["zero"] * 7, "a8_cross_tier": "zero"})
        assert v["code"] == "LABEL-BLIND"
        assert v["conditions"]["b_dead"] is True


def _blk(mean_delta, n_sig):
    return {"rows": [], "n_rows": 5, "mean_delta": mean_delta, "min_delta": mean_delta,
            "max_delta": mean_delta, "n_sig": n_sig,
            "pass": bool(mean_delta >= 0.02 and n_sig >= 4)}


def _run_json(fold, arm, seed, *, acc, within=(0.03, 5), cross=(0.03, 5),
              budget=(0.03, 5), split="S1"):
    return {
        "fold": fold, "split": split, "arm": arm, "seed": seed,
        "test_dataset": fold, "train_datasets": [], "budgets": list(pq.BUDGETS),
        "n_test_pairs_forward": 100, "n_test_pairs_directed": 200,
        "train_seconds": 12.5,
        "normal": {"acc": acc, "acc_forward": acc, "n": 200,
                   "by_budget": {}, "by_kind": {}},
        "within": _blk(*within), "cross": _blk(*cross), "budget": _blk(*budget),
        "zero_E": None, "zero_ctx": None,
    }


def _write_runs(outdir, runs):
    for r in runs:
        d = outdir / "runs" / r["fold"]
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{r['arm']}_s{r['seed']}.json").write_text(
            json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")


class TestAnalyze:
    """手写 run JSON → analyze()：五个判读码 + 预测对账表。"""

    def _a0(self, cross, budget=(0.03, 5), acc=0.80):
        return [_run_json(d, "main", 0, acc=acc, cross=cross, budget=budget)
                for d in pq.FOLD_TARGETS]

    def _a6(self, acc=0.70):
        return [_run_json(d, "cls_b", 0, acc=acc) for d in pq.FOLD_TARGETS]

    def test_e_alive_end_to_end(self, tmp_path):
        _write_runs(tmp_path, self._a0((0.05, 5)) + self._a6())
        res = m33.analyze(tmp_path)
        assert res["verdict"]["code"] == "E-ALIVE"
        assert res["tests"]["n_folds_A0"] == 7

    def test_label_blind_end_to_end(self, tmp_path):
        runs = self._a0((0.001, 0)) + self._a6(acc=0.80)
        runs.append(_run_json(m33.S2_FOLD, "onehot_oracle", 0, acc=0.9,
                              cross=(0.001, 0), split="S2"))
        _write_runs(tmp_path, runs)
        res = m33.analyze(tmp_path)
        assert res["verdict"]["code"] == "LABEL-BLIND"

    def test_e_pathway_end_to_end(self, tmp_path):
        runs = self._a0((0.001, 0)) + self._a6(acc=0.80)
        runs.append(_run_json(m33.S2_FOLD, "onehot_oracle", 0, acc=0.9,
                              cross=(0.05, 5), split="S2"))
        _write_runs(tmp_path, runs)
        res = m33.analyze(tmp_path)
        assert res["verdict"]["code"] == "E-PATHWAY"

    def test_b_dead_end_to_end(self, tmp_path):
        _write_runs(tmp_path, self._a0((0.01, 2), budget=(0.001, 0)))
        res = m33.analyze(tmp_path)
        assert res["verdict"]["code"] == "B-DEAD"

    def test_e_detect_end_to_end(self, tmp_path):
        runs = [_run_json(d, "main", 0, acc=0.80,
                          cross=(0.01, 3) if i < 4 else (0.001, 0))
                for i, d in enumerate(pq.FOLD_TARGETS)]
        _write_runs(tmp_path, runs)
        res = m33.analyze(tmp_path)
        assert res["verdict"]["code"] == "E-DETECT"

    def test_median_seed_by_within(self, tmp_path):
        runs = [_run_json("Handwriting", "main", s, acc=0.8, within=(w, 5))
                for s, w in zip((0, 1, 2), (0.01, 0.09, 0.05))]
        _write_runs(tmp_path, runs)
        res = m33.analyze(tmp_path)
        assert res["per_arm_fold"]["main"]["Handwriting"]["seed"] == 2

    def test_prediction_table(self, tmp_path):
        runs = self._a0((0.05, 5)) + self._a6()
        runs += [_run_json(d, "wo_shared", 0, acc=0.70)
                 for d in ("Handwriting", "UniMiB-SHAR", "Cricket")]
        _write_runs(tmp_path, runs)
        res = m33.analyze(tmp_path)
        preds = res["predictions"]
        assert set(preds) == {f"P{i}" for i in range(1, 11)}
        assert preds["P1"]["hit"] is True          # pooled normal 0.80 ∈ [0.70, 0.85]
        assert preds["P2"]["hit"] is False         # cross 全 pass，不是 zero
        assert preds["P3"]["hit"] is True          # budget pass 7/7
        assert preds["P7"]["hit"] is None          # 34 号脚本
        assert preds["P8"]["hit"] is None
        assert preds["P10"]["n_covered"] == 3      # A4 只跑 3 折

    def test_cost_table(self, tmp_path):
        _write_runs(tmp_path, self._a0((0.05, 5)))
        res = m33.analyze(tmp_path)
        assert res["cost"]["main"]["n_runs"] == 7
        assert res["cost"]["main"]["total_seconds"] == pytest.approx(7 * 12.5)

    def test_p9_from_delta_calibration(self, tmp_path):
        _write_runs(tmp_path, self._a0((0.05, 5)))
        calib = {d: {"60": {"delta": 0.02}, "120": {"delta": 0.03},
                     "290": {"delta": 0.04}} for d in pq.PQ_DATASETS}
        (tmp_path / "delta_calibration.json").write_text(
            json.dumps(calib), encoding="utf-8")
        res = m33.analyze(tmp_path)
        assert res["predictions"]["P9"]["n_datasets_ge"] == 8
        assert res["predictions"]["P9"]["hit"] is True


# ================================================================ 冒烟（端到端）
class TestSmoke:
    """合成缓存 + 合成窗口的最小端到端（Task 5 缓存缺席时的回落路径）。"""

    def test_pipeline_produces_runs_and_analysis(self, tmp_path):
        res = m33.run_smoke(tmp_path, epochs=1,
                            arms=("cls_only", "cls_b", "main"),
                            datasets=("Handwriting", "Cricket"), n_slices=3,
                            n_cls=4, device="cpu", settf_windows=2, t_max=6,
                            folds=("Handwriting",))
        assert (tmp_path / "rprime_cache.jsonl").exists()
        assert (tmp_path / "delta_calibration.json").exists()
        assert (tmp_path / "tasks.json").exists()
        assert (tmp_path / "analysis.json").exists()
        runs = sorted((tmp_path / "runs").glob("*/*.json"))
        assert len(runs) == 3
        models = sorted((tmp_path / "models").glob("*/*.pt"))
        assert len(models) == 3
        assert sorted((tmp_path / "runs").glob("*/*_correct.npz"))
        assert res["verdict"]["code"]

    def test_saved_model_meta_matches_contract(self, tmp_path):
        m33.run_smoke(tmp_path, epochs=1, arms=("main",),
                      datasets=("Handwriting", "Cricket"), n_slices=3, n_cls=4,
                      device="cpu", settf_windows=2, t_max=6, folds=("Handwriting",))
        path = tmp_path / "models" / "Handwriting" / "main_s0.pt"
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert set(ckpt) == {"state_dict", "meta"}
        required = {"arm", "seed", "fold", "task_branch", "ctx_dim", "cls_dim",
                    "embedding_dim", "fz", "nz", "set_encoder", "e_source",
                    "feature_fold", "settf_windows", "settf_t_max", "budgets",
                    "ctx_encoding"}
        assert required <= set(ckpt["meta"])
        assert ckpt["meta"]["e_source"] in m33.E_SOURCES
        assert ckpt["meta"]["feature_fold"] == "Handwriting"
        model, meta = m33.load_pq_model(path)
        assert meta["arm"] == "main"
        assert model.task_branch == "E+ctx"
        assert model.ctx_dim == 3

    def test_s2_model_meta_points_at_all8_features(self, tmp_path):
        m33.run_smoke(tmp_path, epochs=1, arms=("main_s2",),
                      datasets=("Handwriting", "Cricket"), n_slices=3, n_cls=4,
                      device="cpu", settf_windows=2, t_max=6, folds=("Handwriting",))
        ckpt = torch.load(tmp_path / "models" / m33.S2_FOLD / "main_s2_s0.pt",
                          map_location="cpu", weights_only=False)
        # fold="S2" 反推不出目录 → feature_fold 必须显式指向 all8（Task 7 契约）
        assert ckpt["meta"]["fold"] == m33.S2_FOLD
        assert ckpt["meta"]["e_source"] == "fold"
        assert ckpt["meta"]["feature_fold"] == m33.ALL8_FOLD
        run = json.loads((tmp_path / "runs" / m33.S2_FOLD / "main_s2_s0.json")
                         .read_text(encoding="utf-8"))
        assert run["settings"]["feature_fold"] == m33.ALL8_FOLD
        assert run["settings"]["iron_law_6_compliant"] == "s2_oracle"

    def test_reuses_existing_cache_without_features(self, tmp_path):
        """Task 5 冒烟缓存已在盘、特征一个没有 → 仍应跑通（特征合成独立于缓存合成）。"""
        m33.synth_smoke_inputs(tmp_path, datasets=("Handwriting", "Cricket"),
                               n_slices=3, n_cls=4, budgets=pq.BUDGETS)
        assert (tmp_path / "rprime_cache.jsonl").exists()
        assert not (tmp_path / "folds").exists()      # 缓存合成不再产特征
        m33.run_smoke(tmp_path, epochs=1, arms=("main",), device="cpu",
                      settf_windows=2, t_max=6, folds=("Handwriting",))
        assert (tmp_path / "folds" / "Handwriting" / "task_features"
                / "Handwriting_00.npz").exists()
        assert (tmp_path / "runs" / "Handwriting" / "main_s0.json").exists()

    def test_soft_arm_degrades_when_noise_probe_too_thin(self, tmp_path, capsys):
        """噪声探针材料不足（Task 5 冒烟只有 1 切片 × 2 CLS）→ 跳过 A7，不炸整轮。"""
        m33.synth_smoke_inputs(tmp_path, datasets=("Handwriting", "Cricket"),
                               n_slices=3, n_cls=4, budgets=pq.BUDGETS)
        rows = (tmp_path / "noise_probe.jsonl").read_text(
            encoding="utf-8").splitlines()
        (tmp_path / "noise_probe.jsonl").write_text("\n".join(rows[:2]) + "\n",
                                                    encoding="utf-8")
        res = m33.run_smoke(tmp_path, epochs=1, arms=("soft", "cls_only"),
                            device="cpu", settf_windows=2, t_max=6,
                            folds=("Handwriting",))
        assert "[SKIP]" in capsys.readouterr().out
        assert not (tmp_path / "runs" / "Handwriting" / "soft_s0.json").exists()
        assert (tmp_path / "runs" / "Handwriting" / "cls_only_s0.json").exists()
        assert res["verdict"]["code"]

    def test_shared_only_cache_skips_wo_shared_and_continues(self, tmp_path, capsys):
        """32 号冒烟（每切片只有 shared + GGS、零 random）→ A4 训练集为空。

        整轮不得中断：跳过 A4、记 skipped.json、其余臂照跑、analysis 列出。
        """
        m33.synth_smoke_inputs(tmp_path, datasets=("Handwriting", "Cricket"),
                               n_slices=3, n_cls=4, budgets=pq.BUDGETS,
                               origins=("shared",))
        res = m33.run_smoke(tmp_path, epochs=1, arms=("wo_shared", "cls_only"),
                            device="cpu", settf_windows=2, t_max=6,
                            folds=("Handwriting",))
        out = capsys.readouterr().out
        assert "[SKIP] wo_shared/Handwriting:" in out
        assert not (tmp_path / "runs" / "Handwriting" / "wo_shared_s0.json").exists()
        assert (tmp_path / "runs" / "Handwriting" / "cls_only_s0.json").exists()
        skipped = json.loads(
            (tmp_path / "skipped.json").read_text(encoding="utf-8"))["skipped"]
        assert [(r["arm"], r["fold"], r["seed"]) for r in skipped] == [
            ("wo_shared", "Handwriting", 0)]
        assert res["skipped"] == skipped and res["n_skipped"] == 1
        assert res["skipped_arms"] == ["wo_shared"]
        assert "wo_shared" not in res["per_arm_fold"]       # 不伪造条目
        assert "cls_only" in res["per_arm_fold"]

    def test_main_with_empty_records_still_raises(self, tmp_path):
        """主臂是判读表唯一数据源：材料缺席必须当场炸，绝不静默跳过。"""
        m33.synth_smoke_inputs(tmp_path, datasets=("Handwriting", "Cricket"),
                               n_slices=3, n_cls=4, budgets=pq.BUDGETS)
        m33.synth_smoke_features(tmp_path, slices=["Handwriting_00"],
                                 folds=("Handwriting",), settf_windows=2, t_max=6,
                                 nz=256)
        ctx = m33.PQContext(tmp_path, device="cpu", epochs=1, settf_windows=2,
                            smoke=True)
        ctx.slices_by_ds = {"Handwriting": ctx.slices_by_ds["Handwriting"]}  # 训练侧清空
        with pytest.raises(AssertionError, match="主臂"):
            m33.run_arm(ctx, m33.ARMS["main"], "Handwriting", 0)

    def test_non_main_arm_with_empty_records_is_skippable(self, tmp_path):
        m33.synth_smoke_inputs(tmp_path, datasets=("Handwriting", "Cricket"),
                               n_slices=3, n_cls=4, budgets=pq.BUDGETS)
        ctx = m33.PQContext(tmp_path, device="cpu", epochs=1, settf_windows=2,
                            smoke=True)
        ctx.slices_by_ds = {"Handwriting": ctx.slices_by_ds["Handwriting"]}
        with pytest.raises(m33.ArmUnavailable):
            m33.run_arm(ctx, m33.ARMS["cls_only"], "Handwriting", 0)

    def test_soft_unavailable_is_an_arm_unavailable(self):
        assert issubclass(m33.SoftLabelUnavailable, m33.ArmUnavailable)

    def test_successful_rerun_clears_stale_skip_entry(self, tmp_path):
        m33.synth_smoke_inputs(tmp_path, datasets=("Handwriting", "Cricket"),
                               n_slices=3, n_cls=4, budgets=pq.BUDGETS,
                               origins=("shared",))
        m33.run_smoke(tmp_path, epochs=1, arms=("wo_shared",), device="cpu",
                      settf_windows=2, t_max=6, folds=("Handwriting",))
        assert len(m33.load_skipped(tmp_path)) == 1
        # 换成有 random 的缓存重跑同一组合 → 历史 skip 记录必须消失
        (tmp_path / "rprime_cache.jsonl").unlink()
        m33.synth_smoke_inputs(tmp_path, datasets=("Handwriting", "Cricket"),
                               n_slices=3, n_cls=4, budgets=pq.BUDGETS)
        res = m33.run_smoke(tmp_path, epochs=1, arms=("wo_shared",), device="cpu",
                            settf_windows=2, t_max=6, folds=("Handwriting",))
        assert m33.load_skipped(tmp_path) == []
        assert res["n_skipped"] == 0
        assert "wo_shared" in res["per_arm_fold"]

    def test_thin_cache_chain_skips_absent_fold_and_s2(self, tmp_path, capsys):
        """32 号冒烟形状：每数据集 1 个切片、只覆盖部分数据集。

        折靶从缓存推导（不再写死 Handwriting）；1 切片/数据集给不出 S2 划分 →
        两个 S2 臂跳过；within 也因切片不足而缺席。整条链仍须走完并出 analysis。
        """
        m33.synth_smoke_inputs(tmp_path, datasets=("UniMiB-SHAR", "Cricket"),
                               n_slices=1, n_cls=3, budgets=(4, 8, 12))
        res = m33.run_smoke(tmp_path, epochs=1,
                            arms=("main", "cls_b", "main_s2", "onehot_oracle"),
                            device="cpu", settf_windows=2, t_max=6)
        out = capsys.readouterr().out
        assert "[SKIP] main_s2/S2:" in out and "[SKIP] onehot_oracle/S2:" in out
        assert res["skipped_arms"] == ["main_s2", "onehot_oracle"]
        assert all("S2 划分不可用" in r["reason"] for r in res["skipped"])
        # 折靶推导：缓存里的 UniMiB-SHAR / Cricket 都跑了，Handwriting 不该出现
        assert set(res["per_arm_fold"]["main"]) == {"UniMiB-SHAR", "Cricket"}
        assert res["per_arm_fold"]["main"]["Cricket"]["tiers"]["within"] is None

    def test_analysis_carries_iron_law_flag_and_notes(self, tmp_path):
        m33.run_smoke(tmp_path, epochs=1, arms=("main",),
                      datasets=("Handwriting", "Cricket"), n_slices=3, n_cls=4,
                      device="cpu", settf_windows=2, t_max=6, folds=("Handwriting",))
        res = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
        entry = res["per_arm_fold"]["main"]["Handwriting"]
        assert entry["iron_law_6_compliant"] is True
        assert entry["feature_fold"] == "Handwriting"
        assert res["predictions"]["P1"]["note"]
        assert res["predictions"]["P10"]["note"]


# ================================================================ 修复轮：旋钮/闸
def _fake_pq_dataset(n=10, T=20, C=2):
    """PQDataset 的最小替身（_encode_all_slices 只用 .X/.y/.slices）。"""
    from autocls.data.task_slicer import SlicedTask

    class _DS:
        X = np.random.default_rng(0).standard_normal((n, T, C)).astype(np.float32)
        y = np.array([i % 2 for i in range(n)], dtype=np.int64)
        slices = [SlicedTask(task_id="Cricket_00", pool="Cricket", classes=(0, 1),
                             channels=tuple(range(C)),
                             sample_indices=tuple(range(n)))]

    return _DS()


def _smoke_dir(tmp_path, *, datasets=("Handwriting", "Cricket"), n_slices=3,
               n_cls=4, settf_windows=2, t_max=6, folds=("Handwriting",)):
    """最小合成目录（缓存 + δ 表 + 折特征），不训练——闸门测试的公共前置。"""
    m33.synth_smoke_inputs(tmp_path, datasets=datasets, n_slices=n_slices,
                           n_cls=n_cls, budgets=pq.BUDGETS)
    slices = [f"{d}_{i:02d}" for d in datasets for i in range(n_slices)]
    m33.synth_smoke_features(tmp_path, slices=slices, folds=folds,
                             settf_windows=settf_windows, t_max=t_max, nz=256)
    return slices


class TestSettfWindowsKnob:
    """`settf_windows` 是预注册旋钮（100 → 32），绝不允许静默失效。

    旧行为：npz 里有多少窗口就喂多少，改了旋钮却没重跑 --fit-folds 时读到的
    还是老 npz——meta 记 32、实际喂 100，标定出来的墙钟与判读全对不上。
    """

    def test_windows_count_mismatch_is_fatal(self, tmp_path):
        _smoke_dir(tmp_path, settf_windows=2)
        ctx = m33.PQContext(tmp_path, device="cpu", epochs=1, settf_windows=5,
                            smoke=True)
        with pytest.raises(SystemExit) as e:
            ctx.windows(m33.ARMS["main"], "Handwriting", ["Handwriting_00"])
        msg = str(e.value)
        assert "settf_windows" in msg and "5" in msg and "2" in msg
        assert "--fit-folds" in msg          # 可执行的修复动作

    def test_matching_count_passes(self, tmp_path):
        _smoke_dir(tmp_path, settf_windows=2)
        ctx = m33.PQContext(tmp_path, device="cpu", epochs=1, settf_windows=2,
                            smoke=True)
        w = ctx.windows(m33.ARMS["main"], "Handwriting", ["Handwriting_00"])
        assert w["Handwriting_00"].shape[0] == 2      # 绝不截断成 [:n]

    def test_encode_all_slices_reencodes_on_count_change(self, tmp_path,
                                                         tiny_ts2vec, capsys):
        """npz 已在盘但窗口数与本次 sample_num 不符 → 覆盖重编码（而非跳过）。"""
        data = {"Cricket": _fake_pq_dataset()}
        enc = TaskSetEncoder(dim_input=8, nz=8)
        feat = tmp_path / "feat"
        ra = {"embed_seed": 0}
        n = m33._encode_all_slices(feat, data, tiny_ts2vec, enc, ra=ra,
                                   sample_num=3, t_max=10, names=["Cricket"],
                                   allow_unfit=True)
        assert n == 1
        with np.load(feat / "Cricket_00.npz") as z:
            assert z["windows_repr"].shape[0] == 3
        capsys.readouterr()
        m33._encode_all_slices(feat, data, tiny_ts2vec, enc, ra=ra, sample_num=6,
                               t_max=10, names=["Cricket"], allow_unfit=True)
        with np.load(feat / "Cricket_00.npz") as z:
            assert z["windows_repr"].shape[0] == 6     # 覆盖重编码
        assert "重编码" in capsys.readouterr().out

    def test_encode_all_slices_skips_when_count_matches(self, tmp_path,
                                                        tiny_ts2vec):
        data = {"Cricket": _fake_pq_dataset()}
        enc = TaskSetEncoder(dim_input=8, nz=8)
        feat = tmp_path / "feat"
        ra = {"embed_seed": 0}
        m33._encode_all_slices(feat, data, tiny_ts2vec, enc, ra=ra, sample_num=3,
                               t_max=10, names=["Cricket"], allow_unfit=True)
        stamp = (feat / "Cricket_00.npz").stat().st_mtime_ns
        m33._encode_all_slices(feat, data, tiny_ts2vec, enc, ra=ra, sample_num=3,
                               t_max=10, names=["Cricket"], allow_unfit=True)
        assert (feat / "Cricket_00.npz").stat().st_mtime_ns == stamp   # --resume

    def test_capped_slice_is_not_reencoded_forever(self, tmp_path, tiny_ts2vec):
        """实例数 < settf_windows 的切片（源池里 Cricket_02 只有 90 个）。

        encode_task_windows 取 min(sample_num, N)，所以那一格**只可能**有 N 行。
        判据若写死 `== settf_windows`，它会每次重编码、且 windows() 永远判红。
        """
        data = {"Cricket": _fake_pq_dataset(n=10)}
        enc = TaskSetEncoder(dim_input=8, nz=8)
        feat, ra = tmp_path / "feat", {"embed_seed": 0}
        m33._encode_all_slices(feat, data, tiny_ts2vec, enc, ra=ra, sample_num=25,
                               t_max=10, names=["Cricket"], allow_unfit=True)
        with np.load(feat / "Cricket_00.npz") as z:
            assert z["windows_repr"].shape[0] == 10       # 被实例数封顶
        stamp = (feat / "Cricket_00.npz").stat().st_mtime_ns
        m33._encode_all_slices(feat, data, tiny_ts2vec, enc, ra=ra, sample_num=25,
                               t_max=10, names=["Cricket"], allow_unfit=True)
        assert (feat / "Cricket_00.npz").stat().st_mtime_ns == stamp   # 不再重编码
        assert pq.expected_window_count(25, 10) == 10
        assert pq.expected_window_count(25, 90) == 25


class TestPartialCacheGate:
    """33 号绝不在半张真值网格上出判读（Important 3）。"""

    def test_thin_cache_exits_2(self, tmp_path):
        _smoke_dir(tmp_path)
        with pytest.raises(SystemExit) as e:
            m33.PQContext(tmp_path, device="cpu", epochs=1, settf_windows=2)
        assert e.value.code == 2

    def test_missing_list_is_actionable(self, tmp_path, capsys):
        _smoke_dir(tmp_path, datasets=("Handwriting",), n_slices=1, n_cls=4)
        with pytest.raises(SystemExit) as e:
            m33.PQContext(tmp_path, device="cpu", epochs=1, settf_windows=2)
        assert e.value.code == 2
        msg = capsys.readouterr().out
        assert "UniMiB-SHAR" in msg                    # 整个数据集缺
        assert "--allow-partial-cache" in msg
        assert "32_pq_generate_rprime" in msg

    def test_allow_flag_marks_partial(self, tmp_path, monkeypatch):
        _smoke_dir(tmp_path)
        # δ 真源另有其闸（TestDeltaTruthSource）：这里只测满格闸的放行位
        monkeypatch.setattr(m33, "load_delta_source", lambda *a, **k: ({}, "stub"))
        ctx = m33.PQContext(tmp_path, device="cpu", epochs=1, settf_windows=2,
                            allow_partial_cache=True)
        assert ctx.partial_cache is True and ctx.grid_gaps

    def test_smoke_is_exempt(self, tmp_path):
        _smoke_dir(tmp_path)
        ctx = m33.PQContext(tmp_path, device="cpu", epochs=1, settf_windows=2,
                            smoke=True)
        assert ctx.partial_cache is True              # 如实标记，但不拦路

    def test_grid_gaps_on_full_synthetic_grid(self):
        """满格（8 数据集 × 注册表切片 × 每预算 10+10 有限记录）→ 零缺口。"""
        cache = {}
        for d in pq.PQ_DATASETS:
            for s in pq.pq_slice_ids(d):
                for o, n in (("shared", pq.N_SHARED), ("random", pq.N_RANDOM)):
                    for j in range(n):
                        for b in pq.BUDGETS:
                            cache[(s, f"{o}{j}", b)] = {
                                "task_id": s, "cls_key": f"{o}{j}", "iters": b,
                                "origin": o, "r_prime": 0.5}
        assert m33.cache_grid_gaps(cache, pq.BUDGETS) == []
        # 抠掉一格 R'（写成 null）→ 该 (切片, 预算) 立刻少一条
        cache[("Cricket_00", "shared0", 60)]["r_prime"] = None
        gaps = m33.cache_grid_gaps(cache, pq.BUDGETS)
        assert len(gaps) == 1 and "Cricket_00@60" in gaps[0] and "9/10" in gaps[0]

    def test_ggs_rows_do_not_count_toward_the_20(self):
        cache = {}
        for b in pq.BUDGETS:
            for j in range(pq.N_SHARED):
                cache[("Cricket_00", f"s{j}", b)] = {
                    "task_id": "Cricket_00", "cls_key": f"s{j}", "iters": b,
                    "origin": "shared", "r_prime": 0.5}
            cache[("Cricket_00", "ggs", b)] = {
                "task_id": "Cricket_00", "cls_key": "ggs", "iters": b,
                "origin": "ggs", "r_prime": 0.99}
        gaps = m33.cache_grid_gaps(cache, pq.BUDGETS, datasets=("Cricket",))
        assert any("random 0/10" in g for g in gaps)

    def test_analyze_carries_train_datasets_and_flag(self, tmp_path):
        runs = [_run_json(d, "main", 0, acc=0.8) for d in pq.FOLD_TARGETS]
        for r in runs:
            r["train_datasets"] = sorted(pq.fold_split(r["fold"])[0])
            r["partial_cache"] = False
        runs[0]["partial_cache"] = True
        _write_runs(tmp_path, runs)
        res = m33.analyze(tmp_path)
        target = runs[0]["fold"]
        entry = res["per_arm_fold"]["main"][target]
        assert entry["train_datasets"] == sorted(pq.fold_split(target)[0])
        assert target not in entry["train_datasets"]        # 铁律 6 可核对
        assert entry["partial_cache"] is True
        assert res["partial_cache"] is True                 # 任一跑不满格即标记

    def test_analyze_flag_false_on_full_grid(self, tmp_path):
        runs = [_run_json(d, "main", 0, acc=0.8) for d in pq.FOLD_TARGETS]
        for r in runs:
            r["partial_cache"] = False
        _write_runs(tmp_path, runs)
        assert m33.analyze(tmp_path)["partial_cache"] is False


class TestDeltaTruthSource:
    """δ 的唯一真源是 configs/labels.yaml: pq.delta_per_dataset_budget（Minor 6）。"""

    CAL = {"Cricket": {"60": {"delta": 0.03}, "120": {"delta": 0.04}}}

    def _cal(self, tmp_path, payload=None):
        p = tmp_path / "delta_calibration.json"
        p.write_text(json.dumps(payload if payload is not None else self.CAL),
                     encoding="utf-8")
        return p

    def test_empty_labels_table_is_fatal(self, tmp_path):
        with pytest.raises(SystemExit) as e:
            m33.load_delta_source({"delta_per_dataset_budget": {}},
                                  tmp_path / "nope.json")
        msg = str(e.value)
        assert "--calibrate" in msg and "--apply-delta" in msg

    def test_labels_table_is_the_source(self, tmp_path):
        cfg = {"delta_per_dataset_budget": {"Cricket": {60: 0.03, 120: 0.04}}}
        table, src = m33.load_delta_source(cfg, tmp_path / "nope.json")
        assert table == {("Cricket", 60): 0.03, ("Cricket", 120): 0.04}
        assert "labels.yaml" in src

    def test_cross_check_passes_within_1e_9(self, tmp_path):
        cfg = {"delta_per_dataset_budget":
               {"Cricket": {60: 0.03 + 1e-12, 120: 0.04}}}
        table, src = m33.load_delta_source(cfg, self._cal(tmp_path))
        assert table[("Cricket", 60)] == pytest.approx(0.03)
        assert "对账" in src

    def test_mismatch_exits_with_both_values(self, tmp_path):
        cfg = {"delta_per_dataset_budget": {"Cricket": {60: 0.03, 120: 0.05}}}
        with pytest.raises(SystemExit) as e:
            m33.load_delta_source(cfg, self._cal(tmp_path))
        msg = str(e.value)
        assert "0.05" in msg and "0.04" in msg and "Cricket" in msg

    def test_missing_key_is_a_mismatch(self, tmp_path):
        cfg = {"delta_per_dataset_budget": {"Cricket": {60: 0.03}}}
        with pytest.raises(SystemExit) as e:
            m33.load_delta_source(cfg, self._cal(tmp_path))
        assert "Cricket" in str(e.value) and "120" in str(e.value)

    def test_smoke_reads_the_json(self, tmp_path):
        table, src = m33.load_delta_source({"delta_per_dataset_budget": {}},
                                           self._cal(tmp_path), smoke=True)
        assert table == {("Cricket", 60): 0.03, ("Cricket", 120): 0.04}
        assert "smoke" in src

    def test_context_in_smoke_uses_json(self, tmp_path):
        _smoke_dir(tmp_path)
        ctx = m33.PQContext(tmp_path, device="cpu", epochs=1, settf_windows=2,
                            smoke=True)
        assert ctx.calibration[("Handwriting", 60)] == pytest.approx(0.01)
        assert "smoke" in ctx.delta_source


class TestFoldMaterialHasNoTarget:
    """铁律 6 的两条使用点：δ 表与噪声模型都不得含考侧数据集。"""

    def test_delta_map_and_prob_model_exclude_target(self, tmp_path):
        datasets = ("Handwriting", "Cricket", "UniMiB-SHAR")
        _smoke_dir(tmp_path, datasets=datasets, n_slices=3, n_cls=6,
                   folds=("Handwriting",))
        ctx = m33.PQContext(tmp_path, device="cpu", epochs=1, settf_windows=2,
                            smoke=True)
        target = "Handwriting"
        train_ds = [d for d in datasets if d != target]
        train_slices = [s for d in train_ds
                        for s in ctx.slices_by_ds.get(d, [])]
        dmap = ctx.delta_map(train_slices)
        assert dmap and all(pq.dataset_of(t) != target for t in dmap)
        pm = ctx.prob_model(train_ds)
        assert pm.models and all(d != target for d, _b in pm.models)
