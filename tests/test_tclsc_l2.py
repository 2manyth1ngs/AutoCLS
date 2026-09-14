"""WP16-TCLSC-L2：CLS 流水线图 + GIN——先于实现（铁律 5）。协议 results/tclsc_l2/PROTOCOL.md。

A. `autocls/comparator/cls_graph.py`：pipeline_edges（`order` → 增强链边 + 阶段边）、GraphSpec（19 字段 + hub，档位数来自 space.yaml）、
   GraphPacker（438 维：levels ⊕ scalars ⊕ A；与 scalar/binned 编码一致；三种拓扑）、GraphCLSEncoder（式 14；hub/mean 读出；确定性）。
B. `TCLSC(cls_encoder="graph")`：默认 "mlp" 参数名不变；graph 版前向 / 反对称 / `cls_in_dim`。
C. 33 `--encoder-mode graph` + graph 选项 → 冒烟训 cls_only → meta 记 graph_opts；34 loader 重建；56 `encoder_for_meta` / `rank_local_models` 支持 graph。
D. 61 号门与读数码。
零 GPU、不读真实缓存。
"""
from __future__ import annotations

import importlib
import json

import numpy as np
import pytest
import torch

from autocls.comparator.tclsc_model import TCLSC
from autocls.contrastive.augmentations import ORDERS
from autocls.labeling import cls_key
from autocls.space.cls_encoder import CLSEncoder
from autocls.space.cls_space import CLSSpace
from scripts import pq_common as pq

m33 = importlib.import_module("scripts.33_pq_comparator_loo")
g34 = importlib.import_module("scripts.34_pq_search")
g56 = importlib.import_module("scripts.56_tclsc_l1")

SPACE = CLSSpace.from_yaml()
N_FIELDS = len(SPACE.field_names)          # 19
N_NODES = N_FIELDS + 1
PACK_DIM = 2 * N_FIELDS + N_NODES * N_NODES  # 438


def _cg():
    return importlib.import_module("autocls.comparator.cls_graph")


def _g61():
    return importlib.import_module("scripts.61_tclsc_l2")


def _cls(seed=0, **over):
    rng = np.random.default_rng(seed)
    c = SPACE.sample_random(rng)
    return c if not over else type(c)(**{**c.to_dict(), **over})


# ================================================================ A. 图
class TestPipelineEdges:
    def test_aug_chain_follows_orders(self):
        cg = _cg()
        for k, chain in enumerate(ORDERS):
            e = set(map(frozenset, cg.pipeline_edges(k)))
            for a, b in zip(chain, chain[1:]):
                assert frozenset((a, b)) in e
            assert frozenset((chain[-1], "emb_jitter")) in e
        assert set(map(frozenset, cg.pipeline_edges(0))) != set(map(frozenset, cg.pipeline_edges(4)))

    def test_stage_edges_and_validity(self):
        cg = _cg()
        e = set(map(frozenset, cg.pipeline_edges(0)))
        for a, b in (("emb_jitter", "emb_mask"), ("emb_mask", "norm"), ("norm", "sim_func"), ("sim_func", "temperature"),
                     ("temperature", "loss_type")):
            assert frozenset((a, b)) in e
        for f in ("instance", "temporal", "cross_scale", "kernel", "pool", "adj_neighbor"):
            assert frozenset((f, "loss_type")) in e
        names = set(SPACE.field_names) | {cg.HUB}
        assert all(a in names and b in names for a, b in cg.pipeline_edges(0))
        assert all(frozenset((cg.HUB, f)) in e for f in SPACE.field_names)      # 枢纽连全部


class TestGraphPacker:
    def test_dim_and_layout(self):
        cg = _cg()
        p = cg.GraphPacker(SPACE, topology="pipe")
        assert p.mode == "graph" and p.dim == PACK_DIM
        v = p.encode(_cls())
        assert v.shape == (PACK_DIM,) and v.dtype == np.float32
        lv, sc, A = cg.unpack(v, N_FIELDS)
        assert lv.shape == (N_FIELDS,) and sc.shape == (N_FIELDS,) and A.shape == (N_NODES, N_NODES)
        assert np.array_equal(A, A.T) and not np.diag(A).any()

    def test_levels_and_scalars_match_legacy_encoders(self):
        cg = _cg()
        p = cg.GraphPacker(SPACE, topology="pipe")
        c = _cls(3)
        lv, sc, _A = cg.unpack(p.encode(c), N_FIELDS)
        binned, scalar = CLSEncoder(SPACE, mode="binned"), CLSEncoder(SPACE)
        vb, vs = binned.encode(c), scalar.encode(c)
        for i, name in enumerate(SPACE.field_names):
            spec = SPACE.field_spec(name)
            sl = binned.field_slices[name]
            seg = vb[sl]
            if spec["type"] == "bool":
                assert lv[i] == (1 if getattr(c, name) else 0)
            elif seg.any():
                assert lv[i] == int(np.argmax(seg))
            else:
                assert lv[i] == len(spec["choices"])                          # None → 无效档索引
            if spec["type"] in ("numeric", "log_numeric"):
                assert sc[i] == pytest.approx(vs[scalar.field_slices[name]][0])
            else:
                assert sc[i] == 0.0

    def test_topologies(self):
        cg = _cg()
        c = _cls(1)
        _lv, _sc, A_pipe = cg.unpack(cg.GraphPacker(SPACE, topology="pipe").encode(c), N_FIELDS)
        _lv, _sc, A_full = cg.unpack(cg.GraphPacker(SPACE, topology="full").encode(c), N_FIELDS)
        _lv, _sc, A_none = cg.unpack(cg.GraphPacker(SPACE, topology="none").encode(c), N_FIELDS)
        assert A_full.sum() == N_NODES * (N_NODES - 1) and not A_none.any()
        assert 0 < A_pipe.sum() < A_full.sum()
        assert A_pipe[N_NODES - 1].sum() == N_FIELDS                           # hub 行连全部字段
        with pytest.raises(ValueError):
            cg.GraphPacker(SPACE, topology="ring")

    def test_order_changes_edges_only(self):
        cg = _cg()
        p = cg.GraphPacker(SPACE, topology="pipe")
        c0, c4 = _cls(2, order=0), _cls(2, order=4)
        lv0, sc0, A0 = cg.unpack(p.encode(c0), N_FIELDS)
        lv4, sc4, A4 = cg.unpack(p.encode(c4), N_FIELDS)
        i = SPACE.field_names.index("order")
        assert np.array_equal(np.delete(lv0, i), np.delete(lv4, i)) and np.array_equal(sc0, sc4)
        assert not np.array_equal(A0, A4)


class TestGraphCLSEncoder:
    def _enc(self, readout="hub", use_scalar=True, seed=0, d=16):
        cg = _cg()
        torch.manual_seed(seed)
        return cg.GraphCLSEncoder(cg.GraphSpec.from_space(SPACE), d, n_layers=2, readout=readout, use_scalar=use_scalar)

    def _batch(self, topology="pipe", n=5):
        cg = _cg()
        p = cg.GraphPacker(SPACE, topology=topology)
        return torch.as_tensor(np.stack([p.encode(_cls(s)) for s in range(n)]))

    def test_forward_shape_and_determinism(self):
        m = self._enc()
        x = self._batch()
        with torch.no_grad():
            y1, y2 = m(x), m(x)
        assert y1.shape == (5, 16) and torch.equal(y1, y2)
        assert m.in_features == PACK_DIM

    def test_readouts_differ_and_scalar_matters(self):
        x = self._batch()
        with torch.no_grad():
            assert not torch.allclose(self._enc("hub")(x), self._enc("mean")(x))
            a, b = self._enc("hub", True)(x), self._enc("hub", False)(x)
        assert not torch.allclose(a, b)

    def test_no_edges_makes_order_irrelevant(self):
        """topology=none + mean 读出：只有 order 不同的两条 CLS 输出相同（除 order 节点自身的嵌入差）……
        更强的检验：把 order 节点也置同档，则 A 全零下输出逐位相同；pipe 下则不同。"""
        cg = _cg()
        m = self._enc("mean")
        c_a, c_b = _cls(7, order=1), _cls(7, order=3)
        pn = cg.GraphPacker(SPACE, topology="none")
        pp = cg.GraphPacker(SPACE, topology="pipe")
        va, vb = pn.encode(c_a), pn.encode(c_b)
        i = SPACE.field_names.index("order")
        vb[i] = va[i]                                                          # 抹平 order 节点自身的档位差
        with torch.no_grad():
            assert torch.allclose(m(torch.as_tensor(va)[None]), m(torch.as_tensor(vb)[None]))
            wa, wb = pp.encode(c_a), pp.encode(c_b)
            wb[i] = wa[i]
            assert not torch.allclose(m(torch.as_tensor(wa)[None]), m(torch.as_tensor(wb)[None]))


# ================================================================ B. TCLSC
class TestTclscGraph:
    def test_default_mlp_unchanged(self):
        torch.manual_seed(0)
        m = TCLSC(cls_dim=116, set_encoder=None, embedding_dim=8, fz=8, ctx_dim=0, task_branch="none")
        assert m.cls_encoder == "mlp" and m.cls_in_dim == 116
        names = {n.rsplit(".", 1)[0] for n, _ in m.named_parameters()}
        assert names == {"cls_mlp.0", "cls_mlp.2", "cls_fc.0", "pred_fc.0", "pred_fc.2"}

    @pytest.mark.parametrize("head", ["concat", "diff"])
    def test_graph_forward_and_antisymmetry(self, head):
        cg = _cg()
        torch.manual_seed(0)
        m = TCLSC(cls_dim=PACK_DIM, set_encoder=None, embedding_dim=8, fz=8, ctx_dim=0, task_branch="none", head=head,
                  cls_encoder="graph", graph_opts={"topology": "pipe", "readout": "hub", "scalar": True, "gin_layers": 2}).eval()
        assert m.cls_in_dim == PACK_DIM
        p = cg.GraphPacker(SPACE, topology="pipe")
        a = torch.as_tensor(np.stack([p.encode(_cls(s)) for s in range(4)]))
        b = torch.as_tensor(np.stack([p.encode(_cls(s + 10)) for s in range(4)]))
        with torch.no_grad():
            pab, pba = m(a, b, None, None), m(b, a, None, None)
        assert pab.shape == (4,)
        if head == "diff":
            assert torch.allclose(pab + pba, torch.ones(4), atol=1e-6)

    def test_graph_requires_matching_dim(self):
        with pytest.raises(ValueError):
            TCLSC(cls_dim=116, set_encoder=None, embedding_dim=8, fz=8, ctx_dim=0, task_branch="none",
                  cls_encoder="graph", graph_opts={"topology": "pipe", "readout": "hub", "scalar": True, "gin_layers": 2})


# ================================================================ C. 33 / 34 / 56
GRAPH_OPTS = {"topology": "pipe", "readout": "hub", "scalar": True, "gin_layers": 2}


class TestScript33Graph:
    def test_make_encoder_and_parser(self):
        cg = _cg()
        enc = m33.make_cls_encoder(SPACE, "graph", graph_opts=GRAPH_OPTS)
        assert isinstance(enc, cg.GraphPacker) and enc.dim == PACK_DIM and enc.topology == "pipe"
        assert isinstance(m33.make_cls_encoder(SPACE, "binned"), CLSEncoder)
        ap = m33.build_parser()
        a = ap.parse_args(["--train", "--encoder-mode", "graph", "--graph-topology", "full", "--graph-readout", "mean", "--no-graph-scalar"])
        assert a.encoder_mode == "graph" and a.graph_topology == "full" and a.graph_readout == "mean" and a.graph_scalar is False

    def test_train_cls_only_graph_records_opts(self, tmp_path):
        m33.synth_smoke_inputs(tmp_path, datasets=("Handwriting", "Cricket"), n_slices=3, n_cls=4, budgets=pq.BUDGETS)
        m33.cmd_train(tmp_path, arms=["cls_only"], folds=["Handwriting"], seeds=[0], resume=False, device="cpu", epochs=2,
                      settf_windows=2, smoke=True, encoder_mode="graph", graph_opts=GRAPH_OPTS)
        blob = torch.load(tmp_path / "models" / "Handwriting" / "cls_only_s0.pt", map_location="cpu", weights_only=True)
        meta = blob["meta"]
        assert meta["encoder_mode"] == "graph" and meta["cls_dim"] == PACK_DIM and meta["graph_opts"] == GRAPH_OPTS
        run = json.loads((tmp_path / "runs" / "Handwriting" / "cls_only_s0.json").read_text(encoding="utf-8"))
        assert run["settings"]["encoder_mode"] == "graph" and run["settings"]["graph_opts"] == GRAPH_OPTS
        model, meta2 = g34.load_pq_model(tmp_path / "models" / "Handwriting" / "cls_only_s0.pt", expect_fold="Handwriting",
                                         expect_budgets=pq.BUDGETS)
        assert model.cls_encoder == "graph" and model.cls_in_dim == PACK_DIM
        enc = g56.encoder_for_meta(meta2)
        assert enc.mode == "graph" and enc.topology == "pipe"
        rng = np.random.default_rng(7)
        cs = [SPACE.sample_random(rng) for _ in range(5)]
        cat = {"Handwriting_00": [("shared", c, cls_key(c)) for c in cs]}
        out = g56.rank_local_models({"cls_only": [(model, meta2)]}, cat, budgets=pq.BUDGETS)
        assert sorted(out["60"]["Handwriting_00"]["cls_only"]) == sorted(cls_key(c) for c in cs)

    def test_encoder_for_meta_graph_needs_opts(self):
        with pytest.raises(ValueError):
            g56.encoder_for_meta({"encoder_mode": "graph", "cls_dim": PACK_DIM})


# ================================================================ D. 61 号
class TestGatesL2:
    FOLDS = ("HW", "UM", "PS", "UW", "ET", "CR", "HM")

    def _inputs(self, pipe, mlp, full, deltas_pipe_mlp, deltas_pipe_full):
        pooled = {"gin_pipe_hub": {"pair_acc": pipe}, "mlp_binned": {"pair_acc": mlp}, "gin_full_mean": {"pair_acc": full}}
        by_fold = {f: {"gin_pipe_hub": mlp + d1, "mlp_binned": mlp, "gin_full_mean": mlp + d1 - d2}
                   for f, d1, d2 in zip(self.FOLDS, deltas_pipe_mlp, deltas_pipe_full)}
        return pooled, by_fold

    def test_codes(self):
        g = _g61()
        assert g.verdict_l2(g.gates_l2(*self._inputs(0.785, 0.758, 0.770, [0.03] * 7, [0.015] * 7))) == "L2-GRAPH-WIN"
        assert g.verdict_l2(g.gates_l2(*self._inputs(0.785, 0.758, 0.783, [0.03] * 7, [0.002] * 7))) == "L2-EMB-ONLY"
        assert g.verdict_l2(g.gates_l2(*self._inputs(0.765, 0.758, 0.760, [0.007] * 7, [0.005] * 7))) == "L2-FLAT"
        gt = g.gates_l2(*self._inputs(0.785, 0.758, 0.770, [0.03] * 4 + [-0.01] * 3, [0.015] * 7))
        assert not gt["G_L2a"]["pass"]                                        # 折数不够
