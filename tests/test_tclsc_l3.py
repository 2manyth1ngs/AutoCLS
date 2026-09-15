"""WP16-TCLSC-L3：融合头 × 标签形式——先于实现（铁律 5）。协议 results/tclsc_l3/PROTOCOL.md。

A. `TCLSC(head=…)`：concat 默认逐位不变（参数名集合不变）；diff / film 严格反对称、任务向量确实进了打分；
   34 号 loader 按 meta.head 重建。
B. `autocls/comparator/zdiff_labels.zdiff_pairs`：任务内全对、p=σ(k·d)、方向经 better_fn、任务内 z（尺度不变）、k 单调、origins 过滤。
C. `scripts/33`：独立表 `L3_ARMS`（8 臂，S2，3 种子）+ `arm_spec()`；冒烟目录上训 `l3_diff_zdiff` → meta/run json 记 head / label_mode。
D. `scripts/56.rank_local_models(ctx_rows=)`：ctx 臂必须给 ctx，维度互校。
E. `scripts/57_tclsc_l3.py` 纯函数：ctx 行与 33 逐位、跨数据集错位、|Δz| 加权配对 acc、S2 考侧实例、消融 Δ、门与读数码。
零 GPU、不读真实缓存。
"""
from __future__ import annotations

import importlib
import json

import numpy as np
import pytest
import torch

from autocls.comparator.tclsc_engine import ClsRecord
from autocls.comparator.tclsc_model import TCLSC
from autocls.labeling import cls_key
from autocls.space.cls_encoder import CLSEncoder
from autocls.space.cls_space import CLSSpace
from scripts import pq_common as pq

m33 = importlib.import_module("scripts.33_pq_comparator_loo")
g34 = importlib.import_module("scripts.34_pq_search")
g56 = importlib.import_module("scripts.56_tclsc_l1")

SPACE = CLSSpace.from_yaml()
DIM_BINNED = 116
CTX_DIM = len(pq.PQ_DATASETS) + len(pq.BUDGETS)      # 8 + 3
L3_NAMES = ("l3_none_hard", "l3_none_zdiff", "l3_concat_hard", "l3_concat_zdiff",
            "l3_diff_hard", "l3_diff_zdiff", "l3_film_hard", "l3_film_zdiff")


def _g57():
    return importlib.import_module("scripts.57_tclsc_l3")


def _zd():
    return importlib.import_module("autocls.comparator.zdiff_labels")


def _model(head, *, task_branch="ctx", ctx_dim=CTX_DIM, seed=0, cls_dim=10, emb=8, fz=8):
    torch.manual_seed(seed)
    return TCLSC(cls_dim=cls_dim, set_encoder=None, embedding_dim=emb, fz=fz,
                 ctx_dim=ctx_dim, task_branch=task_branch, head=head).eval()


def _xy(seed=1, n=16, d=10):
    rng = np.random.default_rng(seed)
    a = torch.as_tensor(rng.standard_normal((n, d)).astype(np.float32))
    b = torch.as_tensor(rng.standard_normal((n, d)).astype(np.float32))
    ctx = torch.zeros(n, CTX_DIM)
    ctx[:, 2] = 1.0
    ctx[:, len(pq.PQ_DATASETS) + 1] = 1.0
    return a, b, ctx


# ================================================================ A. 头
class TestHeads:
    def test_heads_constant_and_invalid_rejected(self):
        assert TCLSC.HEADS[:3] == ("concat", "diff", "film") and "field" in TCLSC.HEADS      # R 检验加了 field 头
        with pytest.raises(ValueError):
            _model("bilinear")

    def test_default_concat_keeps_parameter_names(self):
        torch.manual_seed(0)
        m = TCLSC(cls_dim=10, set_encoder=None, embedding_dim=8, fz=8, ctx_dim=CTX_DIM, task_branch="ctx")
        assert m.head == "concat"
        names = {n.rsplit(".", 1)[0] for n, _ in m.named_parameters()}
        assert names == {"cls_mlp.0", "cls_mlp.2", "cls_fc.0", "set_fc.0", "pred_fc.0", "pred_fc.2"}

    def test_default_concat_init_order_unchanged(self):
        """同 seed 下 concat 头的初始化必须与改头前逐位相同：RNG 消耗顺序 cls_mlp → cls_fc → set_fc → pred_fc。

        C0 曾因构造顺序变成 set_fc 先于 cls_fc 而失守（bce_first 0.69324 vs 0.69363 → normal acc 差 0.008）。
        """
        torch.manual_seed(0)
        m = TCLSC(cls_dim=10, set_encoder=None, embedding_dim=8, fz=8, ctx_dim=CTX_DIM, task_branch="ctx")
        torch.manual_seed(0)
        ref = [torch.nn.Linear(10, 8), torch.nn.Linear(8, 8), torch.nn.Linear(16, 8),
               torch.nn.Linear(CTX_DIM, 8), torch.nn.Linear(16, 8), torch.nn.Linear(8, 1)]
        got = [m.cls_mlp[0], m.cls_mlp[2], m.cls_fc[0], m.set_fc[0], m.pred_fc[0], m.pred_fc[2]]
        for r, g in zip(ref, got):
            assert torch.equal(r.weight, g.weight) and torch.equal(r.bias, g.bias)

    @pytest.mark.parametrize("head", ["diff", "film"])
    def test_antisymmetric_with_ctx(self, head):
        m = _model(head)
        a, b, ctx = _xy()
        with torch.no_grad():
            p_ab = m(a, b, None, ctx)
            p_ba = m(b, a, None, ctx)
        assert torch.allclose(p_ab + p_ba, torch.ones_like(p_ab), atol=1e-6)
        assert m.antisymmetric is True

    @pytest.mark.parametrize("head", ["diff", "film"])
    def test_antisymmetric_without_task_branch(self, head):
        m = _model(head, task_branch="none", ctx_dim=0)
        a, b, _ = _xy()
        with torch.no_grad():
            assert torch.allclose(m(a, b, None, None) + m(b, a, None, None), torch.ones(a.shape[0]), atol=1e-6)

    def test_concat_is_not_antisymmetric(self):
        m = _model("concat")
        a, b, ctx = _xy()
        with torch.no_grad():
            s = m(a, b, None, ctx) + m(b, a, None, ctx)
        assert m.antisymmetric is False
        assert not torch.allclose(s, torch.ones_like(s), atol=1e-3)

    @pytest.mark.parametrize("head", ["diff", "film"])
    def test_task_vector_changes_output(self, head):
        """任务向量必须进打分：换数据集 one-hot，同一对 (a, b) 的输出要变。"""
        m = _model(head)
        a, b, ctx = _xy()
        ctx2 = ctx.clone()
        ctx2[:, 2], ctx2[:, 5] = 0.0, 1.0
        with torch.no_grad():
            assert not torch.allclose(m(a, b, None, ctx), m(a, b, None, ctx2), atol=1e-6)

    def test_loader_rebuilds_head_from_meta(self, tmp_path):
        spec = m33.ARMS["onehot_oracle"]
        m = _model("film")
        m33.save_model(tmp_path / "x.pt", m, spec, fold="S2", seed=0, cls_dim=10, ctx_dim=CTX_DIM,
                       cfg={"embedding_dim": 8, "fz": 8}, nz=8, settf_windows=2, settf_t_max=6,
                       budgets=pq.BUDGETS, ctx_encoding="onehot", encoder_mode="scalar", head="film")
        model, meta = g34.load_pq_model(tmp_path / "x.pt", expect_fold="S2", expect_budgets=pq.BUDGETS)
        assert meta["head"] == "film" and model.head == "film"
        a, b, ctx = _xy()
        with torch.no_grad():
            assert torch.allclose(model(a, b, None, ctx), m(a, b, None, ctx))

    def test_loader_defaults_missing_head_to_concat(self, tmp_path):
        spec = m33.ARMS["onehot_oracle"]
        m = _model("concat")
        m33.save_model(tmp_path / "x.pt", m, spec, fold="S2", seed=0, cls_dim=10, ctx_dim=CTX_DIM,
                       cfg={"embedding_dim": 8, "fz": 8}, nz=8, settf_windows=2, settf_t_max=6,
                       budgets=pq.BUDGETS, ctx_encoding="onehot", encoder_mode="scalar")
        blob = torch.load(tmp_path / "x.pt", weights_only=True)
        blob["meta"].pop("head", None)                       # 老存档：没有 head 键
        torch.save(blob, tmp_path / "old.pt")
        model, meta = g34.load_pq_model(tmp_path / "old.pt", expect_fold="S2", expect_budgets=pq.BUDGETS)
        assert model.head == "concat"


# ================================================================ B. zdiff 标签
def _recs(task="T", rs=(0.9, 0.8, 0.7, 0.6), origins=None):
    origins = origins or ["shared"] * len(rs)
    return [ClsRecord(task_id=task, key=f"k{i}", origin=o, vec=np.full(3, i, np.float32), r=float(r))
            for i, (r, o) in enumerate(zip(rs, origins))]


def _higher(a, b):
    return a > b


class TestZdiffPairs:
    def test_all_unordered_pairs_per_task(self):
        zd = _zd()
        recs = _recs("A") + _recs("B", rs=(0.5, 0.4, 0.3))
        pairs = zd.zdiff_pairs(recs, _higher, k=2.0, rng=np.random.default_rng(0))
        assert len(pairs) == 6 + 3
        assert all(0.0 < p.p < 1.0 and p.w == 1.0 for p in pairs)
        assert all(p.key_a != p.key_b and p.task_id in ("A", "B") for p in pairs)

    def test_direction_and_hard_label_consistent(self):
        zd = _zd()
        pairs = zd.zdiff_pairs(_recs(), _higher, k=2.0, rng=np.random.default_rng(3))
        for p in pairs:
            assert (p.p > 0.5) == bool(p.y_hard == 1)
            assert (p.p < 0.5) == bool(p.y_hard == 0)

    def test_tie_is_half(self):
        zd = _zd()
        pairs = zd.zdiff_pairs(_recs(rs=(0.7, 0.7, 0.5)), _higher, k=2.0, rng=np.random.default_rng(0))
        tie = [p for p in pairs if {p.key_a, p.key_b} == {"k0", "k1"}][0]
        assert tie.p == 0.5

    def test_within_task_z_is_scale_invariant(self):
        zd = _zd()
        p1 = zd.zdiff_pairs(_recs(), _higher, k=2.0, rng=np.random.default_rng(0))
        p2 = zd.zdiff_pairs(_recs(rs=tuple(3 * r + 0.1 for r in (0.9, 0.8, 0.7, 0.6))), _higher, k=2.0,
                            rng=np.random.default_rng(0))
        assert [round(p.p, 9) for p in p1] == [round(p.p, 9) for p in p2]

    def test_k_sharpens(self):
        zd = _zd()
        soft = zd.zdiff_pairs(_recs(), _higher, k=1.0, rng=np.random.default_rng(0))
        sharp = zd.zdiff_pairs(_recs(), _higher, k=4.0, rng=np.random.default_rng(0))
        assert all(abs(s.p - 0.5) > abs(w.p - 0.5) for s, w in zip(sharp, soft))

    def test_better_fn_defines_direction(self):
        zd = _zd()
        hi = zd.zdiff_pairs(_recs(), _higher, k=2.0, rng=np.random.default_rng(0))
        lo = zd.zdiff_pairs(_recs(), lambda a, b: a < b, k=2.0, rng=np.random.default_rng(0))
        assert [round(p.p, 9) for p in lo] == [round(1 - p.p, 9) for p in hi]

    def test_origins_filter(self):
        zd = _zd()
        recs = _recs(origins=["shared", "shared", "random", "random"])
        pairs = zd.zdiff_pairs(recs, _higher, k=2.0, rng=np.random.default_rng(0), origins=("shared",))
        assert len(pairs) == 1 and {pairs[0].key_a, pairs[0].key_b} == {"k0", "k1"}


# ================================================================ C. 33 号 L3 臂
class TestL3Arms:
    def test_l3_table_shape(self):
        assert set(m33.L3_ARMS) == set(L3_NAMES)
        assert not (set(m33.L3_ARMS) & set(m33.ARMS))            # 不污染 MX-3-PQ 冻结表
        for name, spec in m33.L3_ARMS.items():
            _l3, kind, label = name.split("_")
            assert spec.split == "S2" and spec.seeds == (0, 1, 2)
            assert spec.label_mode == label and spec.encoder is None
            assert spec.iron_law_6_compliant == "s2_oracle"
            if kind == "none":
                assert spec.task_branch == "none" and spec.ctx_kind == "none" and spec.head == "concat"
            else:
                assert spec.task_branch == "ctx" and spec.ctx_kind == "dataset_budget" and spec.head == kind

    def test_arm_spec_resolves_both_tables(self):
        assert m33.arm_spec("main") is m33.ARMS["main"]
        assert m33.arm_spec("l3_diff_zdiff") is m33.L3_ARMS["l3_diff_zdiff"]
        with pytest.raises(KeyError):
            m33.arm_spec("l3_nope")
        assert m33.ZDIFF_K == 2.0

    def test_build_model_passes_head(self):
        spec = m33.L3_ARMS["l3_film_hard"]
        m = m33.build_model(spec, cls_dim=DIM_BINNED, ctx_dim=CTX_DIM, nz=8, dim_input=8,
                            cfg={"embedding_dim": 8, "fz": 8}, seed=0)
        assert m.head == "film" and m.task_branch == "ctx" and m.ctx_dim == CTX_DIM

    def test_train_l3_diff_zdiff_on_smoke_dir(self, tmp_path):
        m33.synth_smoke_inputs(tmp_path, datasets=("Handwriting", "Cricket"), n_slices=3, n_cls=4,
                               budgets=pq.BUDGETS)
        m33.cmd_train(tmp_path, arms=["l3_diff_zdiff"], folds=None, seeds=[0], resume=False,
                      device="cpu", epochs=2, settf_windows=2, smoke=True, encoder_mode="binned")
        blob = torch.load(tmp_path / "models" / m33.S2_FOLD / "l3_diff_zdiff_s0.pt", map_location="cpu",
                          weights_only=True)
        meta = blob["meta"]
        assert meta["head"] == "diff" and meta["encoder_mode"] == "binned"
        assert meta["cls_dim"] == DIM_BINNED and meta["ctx_dim"] == CTX_DIM and meta["task_branch"] == "ctx"
        run = json.loads((tmp_path / "runs" / m33.S2_FOLD / "l3_diff_zdiff_s0.json").read_text(encoding="utf-8"))
        assert run["settings"]["head"] == "diff" and run["settings"]["label_mode"] == "zdiff"
        assert run["settings"]["zdiff_k"] == 2.0
        assert run["cross"] is not None                       # ctx 通道的跨数据集消融照常出块
        model, _ = g34.load_pq_model(tmp_path / "models" / m33.S2_FOLD / "l3_diff_zdiff_s0.pt",
                                     expect_fold=m33.S2_FOLD, expect_budgets=pq.BUDGETS)
        assert model.head == "diff" and model.antisymmetric


# ================================================================ D. 56 号 rank_local_models(ctx_rows=)
def _catalog(n=5, slices=("Handwriting_04", "Handwriting_05")):
    rng = np.random.default_rng(7)
    cs = [SPACE.sample_random(rng) for _ in range(n)]
    return {s: [("shared", c, cls_key(c)) for c in cs] for s in slices}


class TestRankWithCtxRows:
    def test_ctx_model_requires_ctx_rows(self):
        models = {"l3_diff_hard": [(_model("diff", cls_dim=DIM_BINNED),
                                    {"encoder_mode": "binned", "cls_dim": DIM_BINNED, "head": "diff"})]}
        with pytest.raises(ValueError):
            g56.rank_local_models(models, _catalog(), budgets=pq.BUDGETS)

    def test_ctx_rows_used_and_dim_checked(self):
        cat = _catalog()
        models = {"l3_diff_hard": [(_model("diff", cls_dim=DIM_BINNED),
                                    {"encoder_mode": "binned", "cls_dim": DIM_BINNED, "head": "diff"})]}
        rows = {(s, b): np.eye(CTX_DIM, dtype=np.float32)[3] for s in cat for b in pq.BUDGETS}
        out = g56.rank_local_models(models, cat, budgets=pq.BUDGETS, ctx_rows=rows)
        keys = sorted(k for _o, _c, k in cat["Handwriting_04"])
        assert sorted(out["60"]["Handwriting_04"]["l3_diff_hard"]) == keys
        bad = {k: v[:5] for k, v in rows.items()}
        with pytest.raises(ValueError):
            g56.rank_local_models(models, cat, budgets=pq.BUDGETS, ctx_rows=bad)


# ================================================================ E. 57 号纯函数
class TestCtxRows:
    def test_ctx_row_matches_script33(self):
        g = _g57()
        tid = pq.engine_task_id("Handwriting_04", 120)
        ref = m33.context_for(m33.ARMS["onehot_oracle"], [tid], pq.BUDGETS, pq.PQ_DATASETS)[tid]
        assert np.array_equal(g.ctx_row("Handwriting", 120), ref)

    def test_cross_dataset_map_is_derangement_and_deterministic(self):
        g = _g57()
        ds = list(pq.PQ_DATASETS)
        m1, m2 = g.cross_dataset_map(ds, seed=1), g.cross_dataset_map(ds, seed=1)
        assert m1 == m2 and set(m1) == set(ds) and set(m1.values()) == set(ds)
        assert all(k != v for k, v in m1.items())
        assert g.cross_dataset_map(ds, seed=2) != m1


class TestWeightedPairAcc:
    def test_perfect_reversed_and_zero_weight_ties(self):
        g = _g57()
        truth = {"a": 0.9, "b": 0.8, "c": 0.8, "d": 0.6}
        z = {"a": 1.0, "b": 0.3, "c": 0.3, "d": -1.6}
        good = {"a": 3, "b": 2, "c": 1, "d": 0}
        acc, w = g.weighted_pair_acc(good, truth, z)
        assert acc == 1.0 and w == pytest.approx(abs(1.0 - 0.3) * 2 + abs(1.0 + 1.6) + abs(0.3 + 1.6) * 2)
        assert g.weighted_pair_acc({k: -v for k, v in good.items()}, truth, z)[0] == 0.0


def _cat_z_rows(datasets=("Handwriting", "Cricket"), n_slices=6, n=4):
    rng = np.random.default_rng(0)
    rows = []
    for d in datasets:
        for i in range(n_slices):
            for b in pq.BUDGETS:
                for j in range(n):
                    rows.append({"dataset": d, "task_id": f"{d}_{i:02d}", "iters": b, "cls_key": f"k{j}",
                                 "r_prime": 0.9 - 0.1 * j + 0.001 * rng.standard_normal(), "origin": "shared"})
    return g56.zscore_within(rows)


class TestEvalS2:
    def test_only_s2_heldout_instances(self):
        g = _g57()
        cat_z = _cat_z_rows()
        rk = {str(b): {s: {"arm": ["k0", "k1", "k2", "k3"]}
                       for s in ("Handwriting_04", "Handwriting_05", "Cricket_04", "Cricket_05")}
              for b in pq.BUDGETS}
        rows = g.eval_rankings_l3(rk, cat_z, {(d, b): 0.05 for d in ("Handwriting", "Cricket") for b in pq.BUDGETS},
                                  arms=["arm"])
        assert len(rows) == 4 * 3
        assert {r["slice"] for r in rows} == {"Handwriting_04", "Handwriting_05", "Cricket_04", "Cricket_05"}
        m = rows[0]["metrics"]["arm"]
        assert {"pair_acc", "acc_top10", "wacc", "rho"} <= set(m) and m["wacc"] == 1.0 and m["pair_acc"] == 1.0

    def test_s2_test_slices_helper(self):
        g = _g57()
        slices = [f"Handwriting_{i:02d}" for i in range(6)] + [f"SelfRegulationSCP2_{i:02d}" for i in range(3)]
        assert g.s2_test_slices(slices) == ["Handwriting_04", "Handwriting_05", "SelfRegulationSCP2_02"]


class TestAblationAndGates:
    def _rows(self, vals, arm="x", metric="acc_top10"):
        return [{"fold": "S2", "slice": f"s{i}", "budget": 60, "metrics": {arm: {metric: v}}, "regret": {}}
                for i, v in enumerate(vals)]

    def test_ablation_delta_pairs_by_instance(self):
        g = _g57()
        normal = self._rows([0.8, 0.9, 0.7, 0.6, 0.85, 0.75])
        cross = [self._rows([0.7, 0.8, 0.6, 0.5, 0.75, 0.65]), self._rows([0.75, 0.85, 0.65, 0.55, 0.8, 0.7])]
        d = g.ablation_delta(normal, cross, "x", "acc_top10")
        assert d["n"] == 6 and d["mean_delta"] == pytest.approx(0.075) and d["p"] < 0.05
        assert d["wins"] == 6 and d["losses"] == 0

    def _deltas(self, **kw):
        base = {a: {"acc_top10": {"mean_delta": 0.0, "p": 0.9}} for a in L3_NAMES[2:]}
        for a, (d, p) in kw.items():
            base[a] = {"acc_top10": {"mean_delta": d, "p": p}}
        return base

    def test_gates_thresholds(self):
        g = _g57()
        gt = g.gates_l3(self._deltas(l3_diff_hard=(0.031, 0.01)))
        assert gt["G0"]["pass"] and gt["GA"]["pass"] and not gt["GB"]["pass"] and not gt["GC"]["pass"]
        assert not g.gates_l3(self._deltas(l3_diff_hard=(0.031, 0.2)))["GA"]["pass"]       # 不显著
        assert not g.gates_l3(self._deltas(l3_film_hard=(0.02, 0.01)))["GA"]["pass"]       # 量不够
        assert not g.gates_l3(self._deltas(l3_concat_hard=(0.05, 0.001)))["G0"]["pass"]    # A8 未复现

    def test_verdict_codes(self):
        g = _g57()
        def code(**kw):
            return g.verdict_l3(g.gates_l3(self._deltas(**kw)))
        assert code(l3_concat_hard=(0.05, 0.001)) == "L3-ENC-SUFFICES"
        assert code(l3_diff_hard=(0.04, 0.01), l3_diff_zdiff=(0.04, 0.01)) == "L3-HEAD-ONLY"
        assert code(l3_concat_zdiff=(0.04, 0.01), l3_film_zdiff=(0.04, 0.01)) == "L3-LABEL-ONLY"
        assert code(l3_film_zdiff=(0.04, 0.01)) == "L3-HEAD+LABEL"
        assert code() == "L3-FLAT"
