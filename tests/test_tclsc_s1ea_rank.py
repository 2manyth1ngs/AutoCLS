"""WP16-TCLSC 阶段 1 R 检验（低秩字段加性头）——先于实现（铁律 5）。协议 results/tclsc_s1ea_rank/PROTOCOL.md。

A. `tclsc_model.py` field 头：logit = s₀(l_a) − s₀(l_b) + ⟨A·B·t, x_a − x_b⟩（秩 ≤ field_rank）；反对称；diff 头初始化逐位不变；graph 编码拒绝。
B. 33 `ArmSpec.head_rank` / 臂表 / meta 往返（34 加载）；63 臂表 + 端到端；64 `rank_uea(arms=)`；66 门与读数码。
零 GPU、不读真实缓存。
"""
from __future__ import annotations

import importlib
import json

import numpy as np
import pytest
import torch

from autocls.comparator.tclsc_model import HEADS, TCLSC
from autocls.data.mech_features import RESPONSE_PROFILE_COLS
from autocls.space.cls_space import CLSSpace
from scripts import pq_common as pq

m33 = importlib.import_module("scripts.33_pq_comparator_loo")
g34 = importlib.import_module("scripts.34_pq_search")
SPACE = CLSSpace.from_yaml()
EA_COLS = tuple(RESPONSE_PROFILE_COLS)
PHI_COLS = tuple(m33.PHI_COLS)
ALL_COLS = EA_COLS + PHI_COLS
CTX = 5


def _g63():
    return importlib.import_module("scripts.63_tclsc_s1ea")


def _g66():
    return importlib.import_module("scripts.66_tclsc_s1ea_rank")


def _xy(n=6, d=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g), torch.randn(n, d, generator=g), torch.randn(n, CTX, generator=g)


# ================================================================ A. 模型
class TestFieldHead:
    def test_head_registered_and_shapes(self):
        assert "field" in HEADS and "field" in TCLSC.HEADS
        m = TCLSC(cls_dim=10, set_encoder=None, embedding_dim=8, fz=8, ctx_dim=CTX, task_branch="ctx", head="field", field_rank=4)
        assert m.head == "field" and m.field_rank == 4 and m.antisymmetric is True
        assert m.field_w[0].in_features == 8 and m.field_w[0].out_features == 4 and m.field_w[1].out_features == 10
        assert m.field_w[0].bias is None and m.field_w[1].bias is None
        names = {n.rsplit(".", 1)[0] for n, _ in m.named_parameters()}
        assert names == {"cls_mlp.0", "cls_mlp.2", "set_fc.0", "score_mlp.0", "score_mlp.2", "field_w.0", "field_w.1"}
        m2 = TCLSC(cls_dim=10, set_encoder=None, embedding_dim=8, fz=8, ctx_dim=CTX, task_branch="ctx", head="field")
        assert m2.field_rank is None and m2.field_w[0].out_features == 8            # 不限秩 = fz

    def test_forward_decomposition_and_antisymmetry(self):
        m = TCLSC(cls_dim=10, set_encoder=None, embedding_dim=8, fz=8, ctx_dim=CTX, task_branch="ctx", head="field", field_rank=3)
        a, b, ctx = _xy()
        with torch.no_grad():
            p_ab, p_ba = m(a, b, None, ctx), m(b, a, None, ctx)
            assert torch.allclose(p_ab + p_ba, torch.ones_like(p_ab), atol=1e-6)
            t = m.set_fc(ctx)
            s0 = (m.score_mlp(m.cls_mlp(a)) - m.score_mlp(m.cls_mlp(b))).squeeze(1)
            w = m.field_w(t)
            ref = torch.sigmoid(s0 + ((a - b) * w).sum(dim=1))
            assert torch.allclose(p_ab, ref, atol=1e-6)
        # 无任务支路：退化为任务盲加性差
        m0 = TCLSC(cls_dim=10, set_encoder=None, embedding_dim=8, fz=8, ctx_dim=0, task_branch="none", head="field")
        with torch.no_grad():
            p = m0(a, b, None, None)
            assert torch.allclose(p + m0(b, a, None, None), torch.ones_like(p), atol=1e-6)
        assert not hasattr(m0, "field_w")

    def test_other_heads_init_unchanged(self):
        """加 field 头不得改变 diff / concat 头同 seed 的初始化（RNG 消耗顺序）。"""
        torch.manual_seed(0)
        m = TCLSC(cls_dim=10, set_encoder=None, embedding_dim=8, fz=8, ctx_dim=CTX, task_branch="ctx", head="diff")
        torch.manual_seed(0)
        ref = [torch.nn.Linear(10, 8), torch.nn.Linear(8, 8), torch.nn.Linear(CTX, 8), torch.nn.Linear(16, 8), torch.nn.Linear(8, 1)]
        got = [m.cls_mlp[0], m.cls_mlp[2], m.set_fc[0], m.score_mlp[0], m.score_mlp[2]]
        for r, g in zip(ref, got):
            assert torch.equal(r.weight, g.weight) and torch.equal(r.bias, g.bias)

    def test_field_head_rejects_graph_encoder(self):
        from autocls.comparator.cls_graph import GraphSpec
        spec = GraphSpec.from_space(SPACE)
        with pytest.raises(ValueError):
            TCLSC(cls_dim=spec.pack_dim, set_encoder=None, embedding_dim=8, fz=8, ctx_dim=CTX, task_branch="ctx", head="field",
                  cls_encoder="graph", graph_opts={"topology": "pipe", "readout": "hub", "scalar": True, "gin_layers": 1})


# ================================================================ B. 脚本层
def _table(slices, seed=0):
    rng = np.random.default_rng(seed)
    return {s: {"task_id": s, "dataset": pq.dataset_of(s), **{c: float(rng.standard_normal()) for c in ALL_COLS}} for s in slices}


class TestArms33:
    def test_head_rank_and_arms(self):
        assert m33.ArmSpec("x", "S1", "ctx", None, None, "budget").head_rank is None
        a = m33.EA_ARMS
        assert a["fw4_eaphi_hard"].head == "field" and a["fw4_eaphi_hard"].head_rank == 4 and a["fw4_eaphi_hard"].ctx_cols == ALL_COLS
        assert a["fw_eaphi_hard"].head == "field" and a["fw_eaphi_hard"].head_rank is None
        assert m33.arm_spec("fw4_eaphi_hard") is a["fw4_eaphi_hard"] and "fw_eaphi_hard" in m33.CLI_ARMS
        u = _g63().ARMS
        assert u["u_fw4_eaphi_hard"].head == "field" and u["u_fw4_eaphi_hard"].head_rank == 4 and u["u_fw4_eaphi_hard"].ctx_kind == "eaphi_budget"
        assert u["u_fw_eaphi_hard"].head_rank is None
        assert _g63().ctx_cols("eaphi_budget") == ALL_COLS

    def test_smoke_train_field_arm_roundtrip(self, tmp_path):
        m33.synth_smoke_inputs(tmp_path, datasets=("Handwriting", "Cricket"), n_slices=3, n_cls=4, budgets=pq.BUDGETS)
        slices = [f"{d}_{i:02d}" for d in ("Handwriting", "Cricket") for i in range(3)]
        ctx_file = tmp_path / "ea.jsonl"
        with open(ctx_file, "w", encoding="utf-8") as fh:
            for r in _table(slices).values():
                fh.write(json.dumps(r) + "\n")
        m33.cmd_train(tmp_path, arms=["fw4_eaphi_hard"], folds=["Handwriting"], seeds=[0], resume=False, device="cpu",
                      epochs=2, settf_windows=2, smoke=True, encoder_mode="binned", ctx_file=ctx_file)
        p = tmp_path / "models" / "Handwriting" / "fw4_eaphi_hard_s0.pt"
        meta = torch.load(p, map_location="cpu", weights_only=True)["meta"]
        assert meta["head"] == "field" and meta["field_rank"] == 4 and meta["ctx_dim"] == 40
        model, _ = g34.load_pq_model(p, expect_fold="Handwriting", expect_budgets=pq.BUDGETS)
        assert model.head == "field" and model.field_rank == 4 and model.field_w[0].out_features == 4
        run = json.loads((tmp_path / "runs" / "Handwriting" / "fw4_eaphi_hard_s0.json").read_text(encoding="utf-8"))
        assert run["settings"]["head"] == "field" and run["cross"] is not None


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


class TestUcr63:
    def test_train_and_rank_field_arm(self, tmp_path):
        g = _g63()
        rows = _ucr_rows()
        ds = sorted({r["dataset"] for r in rows})
        rng = np.random.default_rng(1)
        ea = {d: {"dataset": d, "task_id": f"{d}_00", **{c: float(rng.standard_normal()) for c in EA_COLS}} for d in ds}
        phi = {d: {"task_id": f"{d}_00", "dataset": d, **{c: float(rng.standard_normal()) for c in PHI_COLS}} for d in ds}
        folds = [["DS02", "DS03"], ["DS00", "DS01"]]
        g.train_fold(rows, ea, phi, folds, 0, arm="u_fw4_eaphi_hard", seed=0, outdir=tmp_path, device="cpu", epochs=2)
        meta = torch.load(tmp_path / "models" / "fold0" / "u_fw4_eaphi_hard_s0.pt", map_location="cpu", weights_only=True)["meta"]
        assert meta["head"] == "field" and meta["field_rank"] == 4 and meta["ctx_kind"] == "eaphi_budget"
        rk = g.rank_fold(rows, ea, phi, folds, 0, arms=["u_fw4_eaphi_hard"], seeds=[0], outdir=tmp_path)
        keys = sorted({r["cls_key"] for r in rows})
        assert sorted(rk["normal"]["60"]["DS02_00"]["u_fw4_eaphi_hard"]) == keys
        assert sorted(rk["zero"]["290"]["DS03_00"]["u_fw4_eaphi_hard"]) == keys and set(rk["cross"]) == {"0", "1", "2"}


class TestScript66:
    def test_arm_lists(self):
        g = _g66()
        assert g.UEA_ARMS == ("fw4_eaphi_hard", "fw_eaphi_hard") and g.UCR_ARMS == ("u_fw4_eaphi_hard", "u_fw_eaphi_hard")
        assert g.UEA_REFS == ("blind_diff_hard", "phi_diff_hard", "eaphi_diff_hard")
        assert g.UCR_REFS == ("u_blind_diff_hard", "u_phi_diff_hard", "u_eaphi_diff_hard")
        assert importlib.import_module("scripts.64_tclsc_s1ea_confirm").rank_uea.__kwdefaults__.get("arms") is None    # 参数化：默认 UEA_ARMS

    def test_gates_and_codes(self):
        g = _g66()
        v = lambda a, b, p: {"mean_a": a, "mean_b": b, "p": p}
        gt = g.gates_rank(v(0.0294, 0.025, 0.01), v(0.025, 0.030, 0.01), v(0.022, 0.019, 0.01))
        assert gt["GR1"]["pass"] and gt["GR2"]["pass"] and gt["GR3"]["pass"] and g.verdict_rank(gt) == "RANK-BOTTLENECK+RETAINED"
        assert g.verdict_rank(g.gates_rank(v(0.0294, 0.025, 0.01), v(0.025, 0.030, 0.01), v(0.022, 0.023, 0.6))) == "RANK-BOTTLENECK-TRADEOFF"
        assert g.verdict_rank(g.gates_rank(v(0.0294, 0.025, 0.01), v(0.025, 0.025, 0.5), v(0.022, 0.019, 0.01))) == "RANK-EXISTENCE"
        assert g.verdict_rank(g.gates_rank(v(0.0294, 0.031, 0.8), v(0.031, 0.031, 0.5), v(0.022, 0.019, 0.01))) == "E-OR-N"
        assert g.verdict_rank(g.gates_rank(v(0.0294, 0.031, 0.8), v(0.031, 0.031, 0.5), v(0.022, 0.023, 0.7))) == "HEAD-TOO-WEAK"
        assert not g.gates_rank(v(0.03, 0.02, 0.01), v(0.03, 0.02, 0.01), v(0.03, 0.02, 0.01))["GR2"]["pass"]      # cross 必须更差
