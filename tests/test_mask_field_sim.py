"""MX-4 WP6 分析脚本（scripts/40_mask_field_sim.py）的纯函数测试。

铁律 5 先测后写。覆盖：S0 配对（protocol_alignment 行按 (dataset, legacy key, seed) 配 none/binomial）、
孪生 z（用 legacy 同组 mean/sd，legacy z 逐位不变）、K1 符号表（官方 Δ 地板过滤）、孪生方向准确率
（对照 always-on / always-off）、interleave / measured 排名、门与读数码、C0 锚（WP1 / WP3 逐位）。
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import pq_common as pq

g40 = importlib.import_module("scripts.40_mask_field_sim")
ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- 合成夹具
def _pa_row(dataset, key, seed, mask, acc, stage="targets", arm="x"):
    return {"dataset": dataset, "cls_key": key, "seed": seed, "stage": stage, "arm": arm,
            "options": {"input_mask": mask}, "acc": acc, "error": None,
            "cls": {"point_mask": 0.2, "emb_mask": 0.0, "crop": 0.5, "freq_mask": 0.0,
                    "resize": 0.0, "rescale": 0.0, "jitter": 0.0, "emb_jitter": 0.0}}


def _src_row(task_id, key, iters, r, twin_of=None, origin="shared"):
    row = {"dataset": task_id.rsplit("_", 1)[0], "task_id": task_id, "cls_key": key,
           "iters": iters, "r_prime": r, "origin": origin, "error": None}
    if twin_of is not None:
        row["twin_of"] = twin_of
    return row


def _deploy_row(dataset, key, accs, twin_of=None, seed=0, stage="s1_catalog"):
    budgets = [50, 100, 200]
    row = {"dataset": dataset, "cls_key": key, "seed": seed, "stage": stage, "error": None,
           "acc": accs[-1], "iters": 200, "checkpoints": budgets,
           "checkpoint_results": [{"iters": b, "acc": a, "elapsed_pretrain_s": 1.0 * i, "eval_seconds": 0.1}
                                  for i, (b, a) in enumerate(zip(budgets, accs))]}
    if twin_of is not None:
        row["twin_of"] = twin_of
    return row


# ---------------------------------------------------------------- S0
def test_s0_pairs_by_legacy_key_and_seed_and_averages_seeds():
    rows = [_pa_row("HW", "k1", 0, "none", 0.40), _pa_row("HW", "k1", 0, "binomial", 0.50),
            _pa_row("HW", "k1", 1, "none", 0.42), _pa_row("HW", "k1", 1, "binomial", 0.48),
            _pa_row("HW", "k2", 0, "none", 0.60),                       # 无 binomial 对 → 不出
            _pa_row("UW", "k1", 0, "binomial", 0.90),                   # 无 none 对 → 不出
            _pa_row("HW", "k3", 0, "none", 0.30), _pa_row("HW", "k3", 0, "binomial", 0.20)]
    out = g40.s0_deltas(rows)
    by = {(d["dataset"], d["cls_key"]): d for d in out}
    assert set(by) == {("HW", "k1"), ("HW", "k3")}
    assert by[("HW", "k1")]["n_seeds"] == 2
    assert by[("HW", "k1")]["delta"] == pytest.approx(((0.50 - 0.40) + (0.48 - 0.42)) / 2)
    assert by[("HW", "k3")]["delta"] == pytest.approx(-0.10)
    assert by[("HW", "k1")]["cls"]["point_mask"] == 0.2
    assert by[("HW", "k1")]["aug_strength"] == pytest.approx(0.2 + 0.5)


def test_s0_ignores_error_rows():
    rows = [_pa_row("HW", "k1", 0, "none", 0.40), {**_pa_row("HW", "k1", 0, "binomial", 0.50), "error": "boom"}]
    assert g40.s0_deltas(rows) == []


# ---------------------------------------------------------------- 孪生 z
def test_twin_z_uses_legacy_group_stats_and_leaves_legacy_z_bitwise():
    g36 = importlib.import_module("scripts.36_prior_sh_sim")
    legacy = [_src_row("HW_00", k, 60, r) for k, r in zip("abcd", [0.1, 0.2, 0.3, 0.4])]
    legacy_z = g36.zscore_within(legacy)
    twins = [_src_row("HW_00", "a_t", 60, 0.5, twin_of="a"), _src_row("HW_00", "b_t", 60, 0.2, twin_of="b"),
             _src_row("HW_01", "c_t", 60, 0.9, twin_of="c")]          # HW_01 无 legacy 组 → 丢弃
    out = g40.twin_z(legacy_z, twins)
    mu, sd = np.mean([0.1, 0.2, 0.3, 0.4]), np.std([0.1, 0.2, 0.3, 0.4])
    assert [t["cls_key"] for t in out] == ["a_t", "b_t"]
    assert out[0]["z"] == pytest.approx((0.5 - mu) / sd)
    assert out[1]["z"] == pytest.approx((0.2 - mu) / sd)
    assert [r["z"] for r in g36.zscore_within(legacy)] == [r["z"] for r in legacy_z]   # legacy 不变
    assert all("z" not in t for t in twins)                                            # 不改入参


# ---------------------------------------------------------------- K1
def test_k1_table_sign_agreement_with_official_floor():
    off = [_deploy_row("HW", "k1", [0.3, 0.4, 0.40]), _deploy_row("HW", "k2", [0.3, 0.4, 0.50]),
           _deploy_row("UW", "k1", [0.8, 0.85, 0.90])]
    on = [_deploy_row("HW", "k1_t", [0.3, 0.4, 0.48], twin_of="k1"),     # Δ_off=+0.08
          _deploy_row("HW", "k2_t", [0.3, 0.4, 0.49], twin_of="k2"),     # Δ_off=−0.01 → 地板下
          _deploy_row("UW", "k1_t", [0.8, 0.85, 0.80], twin_of="k1")]    # Δ_off=−0.10
    legacy = [_src_row("HW_00", "k1", 60, 0.50), _src_row("HW_00", "k1", 120, 0.55),
              _src_row("HW_01", "k1", 60, 0.52),
              _src_row("HW_00", "k2", 60, 0.60),
              _src_row("UW_00", "k1", 60, 0.70)]
    twins = [_src_row("HW_00", "k1_t", 60, 0.56, twin_of="k1"), _src_row("HW_00", "k1_t", 120, 0.60, twin_of="k1"),
             _src_row("HW_01", "k1_t", 60, 0.50, twin_of="k1"),           # 均值 Δ_slice = (+0.06+0.05−0.02)/3 > 0
             _src_row("HW_00", "k2_t", 60, 0.70, twin_of="k2"),
             _src_row("UW_00", "k1_t", 60, 0.75, twin_of="k1")]           # Δ_slice=+0.05 vs Δ_off=−0.10 → 不一致
    t = g40.k1_table(off, on, legacy, twins, floor=0.02, dataset_of=lambda tid: tid.rsplit("_", 1)[0])
    assert t["n_pairs_total"] == 3
    assert t["n_pairs_eligible"] == 2
    assert t["n_agree"] == 1
    assert t["agreement_rate"] == pytest.approx(0.5)
    pairs = {(p["dataset"], p["cls_key"]): p for p in t["pairs"]}
    assert pairs[("HW", "k1")]["delta_official"] == pytest.approx(0.08)
    assert pairs[("HW", "k1")]["delta_slice"] == pytest.approx((0.06 + 0.05 - 0.02) / 3)
    assert pairs[("HW", "k1")]["n_slice_cells"] == 3
    assert pairs[("HW", "k2")]["eligible"] is False


def test_k1_gate_requires_rate_and_n():
    assert g40.k1_pass({"agreement_rate": 0.75, "n_pairs_eligible": 15}) is True
    assert g40.k1_pass({"agreement_rate": 0.75, "n_pairs_eligible": 14}) is False
    assert g40.k1_pass({"agreement_rate": 0.6, "n_pairs_eligible": 30}) is False


# ---------------------------------------------------------------- 孪生方向准确率
def test_twin_direction_accuracy_vs_constant_baselines():
    truth = {"a": 0.50, "a_t": 0.60, "b": 0.50, "b_t": 0.40, "c": 0.50, "c_t": 0.505}
    pairs = [("a", "a_t"), ("b", "b_t"), ("c", "c_t")]
    scores = {"a": 0.0, "a_t": 1.0, "b": 0.0, "b_t": 1.0, "c": 0.0, "c_t": -1.0}   # 预测 a 开好、b 开好（错）
    acc, n = g40.twin_direction_accuracy(scores, truth, pairs, delta=0.02)
    assert n == 2 and acc == pytest.approx(0.5)                     # c 对在 δ 内被剔
    on, _ = g40.twin_direction_accuracy(g40.constant_scores(pairs, +1.0), truth, pairs, delta=0.02)
    off, _ = g40.twin_direction_accuracy(g40.constant_scores(pairs, -1.0), truth, pairs, delta=0.02)
    assert on == pytest.approx(0.5) and off == pytest.approx(0.5)
    acc0, n0 = g40.twin_direction_accuracy(scores, truth, [("c", "c_t")], delta=0.02)
    assert acc0 is None and n0 == 0


# ---------------------------------------------------------------- 排名
def test_interleave_ranking_and_measured_ranking():
    ranked = ["k1", "k2", "k3"]
    twin_map = {"k1": "k1_t", "k3": "k3_t"}
    assert g40.interleave(ranked, twin_map) == ["k1", "k1_t", "k2", "k3", "k3_t"]
    assert g40.all_on(ranked, twin_map) == ["k1_t", "k3_t"]
    assert g40.measured_ranking(ranked, twin_map, top=2) == ["k1", "k1_t", "k2"]
    assert g40.measured_ranking(ranked, twin_map, top=3) == ["k1", "k1_t", "k2", "k3", "k3_t"]


def test_sh12_measured_path_and_cost():
    g36 = importlib.import_module("scripts.36_prior_sh_sim")
    keys = [f"k{i}" for i in range(6)] + [f"k{i}_t" for i in range(6)]
    traj = {k: {50: 0.5 + 0.01 * i, 100: 0.5 + 0.01 * i, 200: 0.5 + 0.01 * i} for i, k in enumerate(keys)}
    twin_map = {f"k{i}": f"k{i}_t" for i in range(6)}
    ranked = g40.measured_ranking([f"k{i}" for i in range(6)], twin_map, top=6)
    assert len(ranked) == 12
    path = g36.sh_path(ranked, traj, (12, 3, 3), (50, 100, 200))
    assert [len(s) for s in path] == [12, 3, 3]
    assert g36.path_cost_iters(path, (50, 100, 200)) == 12 * 50 + 3 * 50 + 3 * 100
    assert g36.sh_pick(path[-1], traj, 200) == "k5_t"


# ---------------------------------------------------------------- 门与读数码
def _per(champ, measured, paper, cost_ratio=0.7):
    return {"champion_mean": champ, "measured_mean": measured, "paper_ts2vec": paper,
            "rung0_cost_ratio": cost_ratio}


def test_gates_k3_k4_and_verdicts():
    per = {d: _per(0.90, 0.905, 0.90) for d in ["A", "B", "C", "D", "E"]}
    per["HW"] = _per(0.43, 0.50, 0.515)
    k2 = {"pass": True}
    g = g40.run_gates_mask(per, k2=k2, k1={"pass": True})
    assert g["K3"]["n_parity"] == 5 and g["K3"]["pass"] is True
    assert g["K4"]["n_within"] == 5 and g["K4"]["pass"] is True
    assert g40.verdict_code_mask(g) == "MASK-WIN"
    g2 = g40.run_gates_mask(per, k2={"pass": False}, k1={"pass": True})
    assert g40.verdict_code_mask(g2) == "MASK-SPACE"
    per["A"]["champion_mean"] = 0.80
    g3 = g40.run_gates_mask(per, k2=k2, k1={"pass": True})
    assert g3["K3"]["pass"] is False and g40.verdict_code_mask(g3) == "MASK-DEAD"
    g4 = g40.run_gates_mask(per, k2=None, k1={"pass": False})
    assert g40.verdict_code_mask(g4).endswith("-NOLEARN")


def test_k4_fails_on_cost_ratio():
    per = {d: _per(0.90, 0.905, 0.90, cost_ratio=0.8) for d in ["A", "B", "C", "D", "E", "F"]}
    g = g40.run_gates_mask(per, k2={"pass": True}, k1={"pass": True})
    assert g["K4"]["pass"] is False


# ---------------------------------------------------------------- C0 锚（真实数据，逐位）
@pytest.mark.skipif(not (ROOT / "results" / "mx4_deploy" / "analysis.json").exists(), reason="缺 WP3 产物")
def test_deploy_c0_matches_wp3_analysis_json():
    wp3 = json.loads((ROOT / "results" / "mx4_deploy" / "analysis.json").read_text(encoding="utf-8"))
    rows = g40.load_wp3_rows()
    for d, ref in wp3["per_target"].items():
        got = g40.deploy_target_off(d, rows)
        assert got["arms"]["SH6-3-3_prior"] == ref["arms"]["SH6-3-3_prior"], d
        assert got["picks"]["SH6-3-3_prior"] == ref["picks"]["SH6-3-3_prior"], d
        assert got["arms"]["prior3"] == ref["arms"]["prior3"], d


@pytest.mark.skipif(not (ROOT / "results" / "mx4_prior_sh" / "analysis.json").exists(), reason="缺 WP1 产物")
def test_source_c0_matches_wp1_per_instance_and_wp2_pooled():
    wp1 = json.loads((ROOT / "results" / "mx4_prior_sh" / "analysis.json").read_text(encoding="utf-8"))
    wp2 = json.loads((ROOT / "results" / "mx4_surrogate" / "analysis.json").read_text(encoding="utf-8"))
    c0 = g40.c0_source(folds=("Handwriting",))
    ref = {(r["slice"], r["budget"]): r for r in wp1["per_instance"] if r["fold"] == "Handwriting"}
    assert len(c0["wp1_rows"]) == len(ref) == 18
    n_sh = 0
    for r in c0["wp1_rows"]:
        e = ref[(r["slice"], r["budget"])]
        assert r["regret"]["prior1"] == e["regret"]["prior1"]
        if "SH6-3-3_prior" in e["regret"]:            # 三级淘汰赛只在 @290 实例上有
            n_sh += 1
            assert r["regret"]["SH6-3-3_prior"] == e["regret"]["SH6-3-3_prior"]
    assert n_sh == 6
    hw_ref = wp2["summary"]["by_fold_pair_acc"]["Handwriting"]["hgb_binned"]
    assert c0["wp2_fold_pair_acc"]["Handwriting"] == pytest.approx(hw_ref, abs=1e-12)
