"""WP16-TCLSC-L1：T-CLSC CLS 支路的 编码 × 目标函数 分解——先于实现（铁律 5）。

协议：results/tclsc_l1/PROTOCOL.md。两部分：

A. `scripts/33_pq_comparator_loo.py` 新增 `--encoder-mode {scalar,binned}`：
   - 默认 scalar 逐位不变（记录向量与 `CLSEncoder(space)` 相同、meta 显式记 "scalar"）；
   - binned 时记录向量 116 维、模型第一层 in_features=116、run json / 模型 meta 都记模式。
B. `scripts/56_tclsc_l1.py` 的纯函数：
   - meta → 打分用编码器（模式与 cls_dim 互校）；
   - rankings → WP2 同口径指标；C0 硬门；
   - MLPZ 回归臂（同 cls_mlp 形状、确定性、能拟合非单调档位效应）；LODO 划分；
   - 本地模型排名（按 meta 模式取向量）；变体目录准备（只复制输入、幂等）；
   - 门 G1/G2/G3/G3′ 与读数码。
零 GPU、不读真实缓存。
"""
from __future__ import annotations

import importlib
import json

import numpy as np
import pytest
import torch

from autocls.comparator.tclsc_model import TCLSC
from autocls.labeling import cls_key
from autocls.space.cls_encoder import CLSEncoder
from autocls.space.cls_space import CLS, CLSSpace
from scripts import pq_common as pq

m33 = importlib.import_module("scripts.33_pq_comparator_loo")
g34 = importlib.import_module("scripts.34_pq_search")

SPACE = CLSSpace.from_yaml()
DIM_SCALAR, DIM_BINNED = 32, 116


def _g56():
    """56 号按测试逐个导入：缺席时每个用例各自红，不拖累 A 部分。"""
    return importlib.import_module("scripts.56_tclsc_l1")


def _smoke_dir(tmp_path, *, datasets=("Handwriting", "Cricket"), n_slices=3,
               n_cls=4, settf_windows=2, t_max=6, folds=("Handwriting",)):
    m33.synth_smoke_inputs(tmp_path, datasets=datasets, n_slices=n_slices,
                           n_cls=n_cls, budgets=pq.BUDGETS)
    slices = [f"{d}_{i:02d}" for d in datasets for i in range(n_slices)]
    m33.synth_smoke_features(tmp_path, slices=slices, folds=folds,
                             settf_windows=settf_windows, t_max=t_max, nz=256)
    return slices


# ================================================================ A. 33 号 --encoder-mode
class TestEncoderModeFlag:
    def test_parser_default_scalar_and_rejects_unknown(self):
        ap = m33.build_parser()
        assert ap.parse_args(["--train"]).encoder_mode == "scalar"
        assert ap.parse_args(["--train", "--encoder-mode", "binned"]).encoder_mode == "binned"
        with pytest.raises(SystemExit):
            ap.parse_args(["--train", "--encoder-mode", "hyper"])      # graph 自 L2 起合法，改用真未知值

    def test_context_binned_records_are_116_dim(self, tmp_path):
        slices = _smoke_dir(tmp_path)
        ctx = m33.PQContext(tmp_path, device="cpu", epochs=1, settf_windows=2,
                            smoke=True, encoder_mode="binned")
        assert ctx.encoder.mode == "binned" and ctx.encoder_mode == "binned"
        recs = ctx.records(slices)
        assert recs and all(r.vec.shape == (DIM_BINNED,) for r in recs)
        # 每个有效字段恰一个 1（one-hot），无效档全零；不会出现标量分量
        for r in recs[:8]:
            assert set(np.unique(r.vec)).issubset({0.0, 1.0})

    def test_context_default_scalar_bitwise_unchanged(self, tmp_path):
        slices = _smoke_dir(tmp_path)
        ctx = m33.PQContext(tmp_path, device="cpu", epochs=1, settf_windows=2, smoke=True)
        assert ctx.encoder_mode == "scalar"
        ref = CLSEncoder(SPACE)
        recs = ctx.records(slices)
        assert recs and all(r.vec.shape == (DIM_SCALAR,) for r in recs)
        for r in recs[:8]:
            sid, b = pq.split_engine_task_id(r.task_id)
            cached = ctx.cache[(sid, r.key, int(b))]
            assert np.array_equal(r.vec, ref.encode(CLS(**cached["cls"])))

    def test_train_cls_only_binned_records_mode_in_meta_and_run_json(self, tmp_path):
        _smoke_dir(tmp_path)
        m33.cmd_train(tmp_path, arms=["cls_only"], folds=["Handwriting"], seeds=[0],
                      resume=False, device="cpu", epochs=2, settf_windows=2,
                      smoke=True, encoder_mode="binned")
        blob = torch.load(tmp_path / "models" / "Handwriting" / "cls_only_s0.pt",
                          map_location="cpu", weights_only=True)
        meta = blob["meta"]
        assert meta["encoder_mode"] == "binned" and meta["cls_dim"] == DIM_BINNED
        assert tuple(blob["state_dict"]["cls_mlp.0.weight"].shape) == (meta["embedding_dim"], DIM_BINNED)
        run = json.loads((tmp_path / "runs" / "Handwriting" / "cls_only_s0.json").read_text(encoding="utf-8"))
        assert run["settings"]["encoder_mode"] == "binned"
        # 34 号的 loader 只凭 meta 就能重建 116 维模型
        model, meta2 = g34.load_pq_model(tmp_path / "models" / "Handwriting" / "cls_only_s0.pt",
                                         expect_fold="Handwriting", expect_budgets=pq.BUDGETS)
        assert meta2["encoder_mode"] == "binned"
        assert model.cls_mlp[0].in_features == DIM_BINNED

    def test_train_default_records_scalar_explicitly(self, tmp_path):
        _smoke_dir(tmp_path)
        m33.cmd_train(tmp_path, arms=["cls_only"], folds=["Handwriting"], seeds=[0],
                      resume=False, device="cpu", epochs=2, settf_windows=2, smoke=True)
        meta = torch.load(tmp_path / "models" / "Handwriting" / "cls_only_s0.pt",
                          map_location="cpu", weights_only=True)["meta"]
        assert meta["encoder_mode"] == "scalar" and meta["cls_dim"] == DIM_SCALAR
        run = json.loads((tmp_path / "runs" / "Handwriting" / "cls_only_s0.json").read_text(encoding="utf-8"))
        assert run["settings"]["encoder_mode"] == "scalar"


# ================================================================ B. 56 号纯函数
class TestEncoderForMeta:
    def test_mode_key_selects_encoder(self):
        g = _g56()
        assert g.encoder_for_meta({"encoder_mode": "binned", "cls_dim": DIM_BINNED}).mode == "binned"
        assert g.encoder_for_meta({"encoder_mode": "scalar", "cls_dim": DIM_SCALAR}).mode == "scalar"

    def test_legacy_meta_without_key_is_scalar(self):
        # pod 上的老存档没有 encoder_mode 键 = scalar（32 维），不得猜成 binned
        assert _g56().encoder_for_meta({"cls_dim": DIM_SCALAR}).mode == "scalar"

    def test_mode_dim_mismatch_raises(self):
        g = _g56()
        with pytest.raises(ValueError):
            g.encoder_for_meta({"encoder_mode": "binned", "cls_dim": DIM_SCALAR})
        with pytest.raises(ValueError):
            g.encoder_for_meta({"cls_dim": DIM_BINNED})          # 无键却是 116 维：不猜


def _cat_rows(dataset="D", slices=("D_00",), budgets=(60,), keys=("a", "b", "c", "d"),
              r_of=lambda i: 0.9 - 0.1 * i):
    rows = []
    for s in slices:
        for b in budgets:
            for i, k in enumerate(keys):
                rows.append({"dataset": dataset, "task_id": s, "iters": b, "cls_key": k,
                             "r_prime": float(r_of(i)), "origin": "shared"})
    return rows


class TestEvalRankings:
    def test_perfect_ranking_scores_one_and_zero_regret(self):
        g = _g56()
        rows = _cat_rows()
        rankings = {"D": {"60": {"D_00": {"main": ["a", "b", "c", "d"]}}}}
        out = g.eval_rankings(rankings, rows, {("D", 60): 0.05}, arms=["main"])
        assert len(out) == 1
        m = out[0]["metrics"]["main"]
        assert m["pair_acc"] == 1.0 and m["n_pairs"] == 6
        assert out[0]["regret"]["main1"] == 0.0 and out[0]["regret"]["main3"] == 0.0
        assert out[0]["fold"] == "D" and out[0]["slice"] == "D_00" and out[0]["budget"] == 60

    def test_reversed_ranking_scores_zero(self):
        g = _g56()
        out = g.eval_rankings({"D": {"60": {"D_00": {"main": ["d", "c", "b", "a"]}}}},
                              _cat_rows(), {("D", 60): 0.05}, arms=["main"])
        assert out[0]["metrics"]["main"]["pair_acc"] == 0.0
        assert out[0]["regret"]["main1"] == pytest.approx(0.3)

    def test_missing_arm_in_instance_is_skipped_not_faked(self):
        g = _g56()
        rankings = {"D": {"60": {"D_00": {"main": ["a", "b", "c", "d"]}}}}
        out = g.eval_rankings(rankings, _cat_rows(), {("D", 60): 0.05}, arms=["main", "cls_only"])
        assert "cls_only" not in out[0]["metrics"]


class TestC0Check:
    def _rows(self, accs, regrets):
        return [{"metrics": {"main": {"pair_acc": a}}, "regret": {"main1": r}}
                for a, r in zip(accs, regrets)]

    def test_within_tolerance_passes(self):
        g = _g56()
        res = g.c0_check(self._rows([0.735, 0.7352], [0.074, 0.0741]))
        assert res["pass"] and res["pair_acc"] == pytest.approx(0.7351)

    def test_outside_tolerance_fails(self):
        g = _g56()
        assert not g.c0_check(self._rows([0.74, 0.74], [0.074, 0.074]))["pass"]
        assert not g.c0_check(self._rows([0.735, 0.735], [0.08, 0.08]))["pass"]


class TestMLPZ:
    def test_trunk_mirrors_cls_mlp_shape(self):
        g = _g56()
        m = g.MLPZ(in_dim=33, hidden=128, seed=0)
        lin = [l for l in m.trunk if isinstance(l, torch.nn.Linear)]
        assert [(l.in_features, l.out_features) for l in lin] == [(33, 128), (128, 128)]
        assert m.head.out_features == 1

    def test_deterministic_given_seed(self):
        g = _g56()
        rng = np.random.default_rng(0)
        X = rng.standard_normal((64, 10)).astype(np.float32)
        y = rng.standard_normal(64).astype(np.float32)
        p1 = g.MLPZ(in_dim=10, seed=3, epochs=20).fit(X, y).predict(X)
        p2 = g.MLPZ(in_dim=10, seed=3, epochs=20).fit(X, y).predict(X)
        assert np.array_equal(p1, p2)
        p3 = g.MLPZ(in_dim=10, seed=4, epochs=20).fit(X, y).predict(X)
        assert not np.array_equal(p1, p3)

    def test_fits_nonmonotone_binned_levels(self):
        """11 档 one-hot、档位效应非单调（crop 0.2 与 0.5 差一个 z 的形状）——能力测试。"""
        g = _g56()
        rng = np.random.default_rng(1)
        table = np.array([0.0, 0.8, -0.5, 0.9, -0.9, 0.3, -0.2, 0.7, -0.6, 0.1, -0.8])
        lv = rng.integers(0, 11, size=400)
        X = np.eye(11, dtype=np.float32)[lv]
        y = (table[lv] + 0.02 * rng.standard_normal(400)).astype(np.float32)
        pred = g.MLPZ(in_dim=11, seed=0, epochs=300).fit(X, y).predict(X)
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        assert r2 >= 0.9


class TestLODO:
    def test_split_excludes_target_from_train(self):
        g = _g56()
        rows = (_cat_rows("A", ("A_00",)) + _cat_rows("B", ("B_00",)) + _cat_rows("T", ("T_00", "T_01")))
        train, test = g.lodo_split(rows, "T")
        assert {r["dataset"] for r in train} == {"A", "B"}
        assert {r["dataset"] for r in test} == {"T"} and len(test) == 8

    def test_fold_eval_mlpz_returns_wp2_shaped_rows(self):
        """合成 3 数据集 × 2 切片 × 3 预算 × 6 CLS：两臂都出 WP2 形状的实例行。"""
        g = _g56()
        rng = np.random.default_rng(0)
        cands = [SPACE.sample_random(rng) for _ in range(6)]
        rows = []
        for d in ("A", "B", "T"):
            for s in (f"{d}_00", f"{d}_01"):
                for b in pq.BUDGETS:
                    for c in cands:
                        r = 0.5 + 0.1 * float(c.temperature <= 1.0) + 0.01 * rng.standard_normal()
                        rows.append({"dataset": d, "task_id": s, "iters": b, "cls_key": cls_key(c),
                                     "cls": c.to_dict(), "r_prime": r, "origin": "shared"})
        cat_z = g.zscore_within(rows)
        enc = {"scalar": CLSEncoder(SPACE), "binned": CLSEncoder(SPACE, mode="binned")}
        delta = {("T", b): 0.02 for b in pq.BUDGETS}
        out = g.fold_eval_mlpz("T", cat_z, delta, enc, seeds=(0,), epochs=30)
        assert len(out) == 6
        for r in out:
            assert r["fold"] == "T" and set(r["metrics"]) == {"mlp_z_scalar", "mlp_z_binned"}
            for arm in ("mlp_z_scalar", "mlp_z_binned"):
                assert {"pair_acc", "n_pairs", "acc_top10", "rho"} <= set(r["metrics"][arm])
                assert arm + "1" in r["regret"] and arm + "3" in r["regret"]


def _toy_model(cls_dim, seed):
    torch.manual_seed(seed)
    return TCLSC(cls_dim=cls_dim, set_encoder=None, embedding_dim=8, fz=8,
                 ctx_dim=0, task_branch="none").eval()


class TestRankLocalModels:
    def _catalog(self, n=5):
        rng = np.random.default_rng(7)
        cs = [SPACE.sample_random(rng) for _ in range(n)]
        return {"T_00": [("shared", c, cls_key(c)) for c in cs],
                "T_01": [("shared", c, cls_key(c)) for c in cs]}

    def test_uses_meta_mode_and_returns_permutations(self):
        g = _g56()
        cat = self._catalog()
        models = {"cls_only": [(_toy_model(DIM_BINNED, 0), {"encoder_mode": "binned", "cls_dim": DIM_BINNED}),
                               (_toy_model(DIM_BINNED, 1), {"encoder_mode": "binned", "cls_dim": DIM_BINNED})]}
        r1 = g.rank_local_models(models, cat, budgets=pq.BUDGETS)
        r2 = g.rank_local_models(models, cat, budgets=pq.BUDGETS)
        assert r1 == r2                                            # 逐位确定
        keys = sorted(k for _o, _c, k in cat["T_00"])
        for b in pq.BUDGETS:
            for s in cat:
                assert sorted(r1[str(b)][s]["cls_only"]) == keys

    def test_scalar_legacy_meta_works(self):
        g = _g56()
        models = {"cls_only": [(_toy_model(DIM_SCALAR, 0), {"cls_dim": DIM_SCALAR})]}
        out = g.rank_local_models(models, self._catalog(), budgets=(60,))
        assert set(out["60"]) == {"T_00", "T_01"}

    def test_mode_dim_mismatch_raises_before_scoring(self):
        g = _g56()
        models = {"cls_only": [(_toy_model(DIM_SCALAR, 0), {"encoder_mode": "binned", "cls_dim": DIM_SCALAR})]}
        with pytest.raises(ValueError):
            g.rank_local_models(models, self._catalog(), budgets=(60,))


class TestPrepareVariantDir:
    def test_copies_only_inputs_and_is_idempotent(self, tmp_path):
        g = _g56()
        src, dst = tmp_path / "src", tmp_path / "dst"
        src.mkdir()
        for name in g.VARIANT_INPUTS:
            (src / name).write_text("x", encoding="utf-8")
        (src / "rankings.json").write_text("{}", encoding="utf-8")     # 产物，不该带过去
        (src / "runs").mkdir()
        copied = g.prepare_variant_dir(src, dst)
        assert sorted(copied) == sorted(g.VARIANT_INPUTS)
        assert not (dst / "rankings.json").exists() and not (dst / "runs").exists()
        stamps = {n: (dst / n).stat().st_mtime_ns for n in g.VARIANT_INPUTS}
        assert g.prepare_variant_dir(src, dst) == []
        assert {n: (dst / n).stat().st_mtime_ns for n in g.VARIANT_INPUTS} == stamps

    def test_missing_required_input_raises(self, tmp_path):
        g = _g56()
        (tmp_path / "src").mkdir()
        with pytest.raises(FileNotFoundError):
            g.prepare_variant_dir(tmp_path / "src", tmp_path / "dst")


class TestGates:
    FOLDS = ("HW", "UM", "PS", "UW", "ET", "CR", "HM")

    def _by_fold(self, scalar, binned_delta):
        return {f: {"cls_only_scalar": scalar, "cls_only_binned": scalar + d}
                for f, d in zip(self.FOLDS, binned_delta)}

    def _pooled(self, **acc):
        return {arm: {"pair_acc": v} for arm, v in acc.items()}

    def test_g1_requires_pooled_and_fold_majority(self):
        g = _g56()
        pooled = self._pooled(cls_only_scalar=0.72, cls_only_binned=0.745, mlp_z_binned=0.70, mlp_z_scalar=0.70)
        ok = g.gates(pooled, self._by_fold(0.72, [0.03] * 6 + [-0.01]))
        assert ok["G1"]["pass"] and ok["G1"]["n_folds_better"] == 6
        few = g.gates(pooled, self._by_fold(0.72, [0.05] * 4 + [-0.01] * 3))
        assert not few["G1"]["pass"]
        small = g.gates(self._pooled(cls_only_scalar=0.72, cls_only_binned=0.735, mlp_z_binned=0.70, mlp_z_scalar=0.70),
                        self._by_fold(0.72, [0.015] * 7))
        assert not small["G1"]["pass"]

    def test_g3_threshold_and_g3prime(self):
        g = _g56()
        pooled = self._pooled(cls_only_scalar=0.72, cls_only_binned=0.72, mlp_z_binned=0.766, mlp_z_scalar=0.74)
        out = g.gates(pooled, self._by_fold(0.72, [0.0] * 7))
        assert out["G3"]["pass"] and out["G3p"]["pass"]
        out2 = g.gates(self._pooled(cls_only_scalar=0.72, cls_only_binned=0.72, mlp_z_binned=0.764, mlp_z_scalar=0.75),
                       self._by_fold(0.72, [0.0] * 7))
        assert not out2["G3"]["pass"] and not out2["G3p"]["pass"]

    def test_g2_na_without_main_binned(self):
        g = _g56()
        pooled = self._pooled(cls_only_scalar=0.72, cls_only_binned=0.72, mlp_z_binned=0.70, mlp_z_scalar=0.70)
        out = g.gates(pooled, self._by_fold(0.72, [0.0] * 7))
        assert out["G2"]["pass"] is None
        pooled["main_binned"] = {"pair_acc": 0.76}
        by_fold = {f: dict(v, main_binned=0.76) for f, v in self._by_fold(0.72, [0.0] * 7).items()}
        assert g.gates(pooled, by_fold)["G2"]["pass"] is True     # 0.76 − 0.735 ≥ 0.02 且 7/7 折

    def test_verdict_codes(self):
        g = _g56()
        def gt(g1, g3):
            return {"G1": {"pass": g1}, "G3": {"pass": g3}, "G2": {"pass": None}, "G3p": {"pass": False}}
        assert g.verdict_code(gt(True, True)) == "L1-ENC+OBJ"
        assert g.verdict_code(gt(True, False)) == "L1-ENC-ONLY"
        assert g.verdict_code(gt(False, True)) == "L1-OBJ-ONLY"
        assert g.verdict_code(gt(False, False)) == "L1-FLAT"
