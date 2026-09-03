"""MX-4 WP2：字段结构化代理 LODO（scripts/37_surrogate_loo.py）测试——先于实现（铁律 5）。

协议：results/mx4_surrogate/PROTOCOL.md。覆盖纯函数：
- δ 存活配对准确率（含 top-k 内口径）；
- 名次 → 得分（T-CLSC 对照臂同口径）；
- 分组置换重要性：零效应字段 ≈ 0、强效应字段大；
- 进化行的 z 归一（用目录 mean/std）与 LCB；
- 门 S1–S3 与读数码。
零 GPU、不读真实缓存。
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from autocls.search.surrogate import FieldSurrogate
from autocls.space.cls_encoder import CLSEncoder
from autocls.space.cls_space import CLSSpace

g37 = importlib.import_module("scripts.37_surrogate_loo")


# ---------------------------------------------------------------- 配对准确率
def test_pair_accuracy_perfect_reversed_and_delta_filter():
    truth = {"a": 0.9, "b": 0.7, "c": 0.5, "d": 0.49}
    good = {"a": 3, "b": 2, "c": 1, "d": 0}
    acc, n = g37.pair_accuracy(good, truth, delta=0.1)
    assert acc == 1.0 and n == 5                     # (c,d) 差 0.01 < δ 被剔除
    bad = {k: -v for k, v in good.items()}
    assert g37.pair_accuracy(bad, truth, delta=0.1)[0] == 0.0
    assert g37.pair_accuracy(good, truth, delta=1.0) == (None, 0)   # 无存活对 → None


def test_pair_accuracy_topk_restricts_to_true_top():
    truth = {"a": 0.9, "b": 0.8, "c": 0.2, "d": 0.1}
    score = {"a": 0, "b": 1, "c": 3, "d": 2}         # top-2 内反序，跨半区也反，只有 (c,d) 对
    acc_all, n_all = g37.pair_accuracy(score, truth, delta=0.05)
    acc_top, n_top = g37.pair_accuracy(score, truth, delta=0.05, top_k=2)
    assert n_top == 1 and acc_top == 0.0
    assert n_all == 6 and acc_all == pytest.approx(1 / 6)


def test_scores_from_ranking_descending():
    s = g37.scores_from_ranking(["b", "a", "c"])
    assert s["b"] > s["a"] > s["c"]


# ---------------------------------------------------------------- 分组置换重要性
def test_grouped_permutation_importance_separates_active_and_inert_fields():
    space = CLSSpace.from_yaml()
    enc = CLSEncoder(space, mode="binned")
    rng = np.random.default_rng(0)
    X = np.stack([enc.encode(space.sample_random(rng)) for _ in range(800)])
    sl = enc.field_slices
    w = np.zeros(X.shape[1])
    w[sl["temperature"]] = np.linspace(2.0, -2.0, sl["temperature"].stop - sl["temperature"].start)
    y = X @ w + rng.normal(scale=0.05, size=X.shape[0])
    m = FieldSurrogate(kind="ridge").fit(X, y)
    imp = g37.grouped_permutation_importance(m, X, y, sl, seed=0)
    assert imp["temperature"] > 10 * max(imp["pool"], imp["loss_type"], 1e-9)
    assert abs(imp["pool"]) < 0.05


# ---------------------------------------------------------------- 进化行 / LCB
def test_instance_z_uses_catalog_moments():
    cat = np.array([0.2, 0.4, 0.6])
    assert g37.instance_z(0.4, cat) == pytest.approx(0.0)
    assert g37.instance_z(0.6, cat) == pytest.approx(1.0 / np.std(cat) * 0.2)


def test_lcb_lowers_uncertain_scores():
    mean = np.array([1.0, 1.0])
    std = np.array([0.0, 0.5])
    out = g37.lcb(mean, std, kappa=1.0)
    assert out[0] == pytest.approx(1.0) and out[1] == pytest.approx(0.5)


# ---------------------------------------------------------------- 门
def test_gates_and_verdict():
    folds = {f: {"hgb_binned": 0.80, "tclsc": 0.72} for f in "ABCDEFG"}
    folds["G"]["hgb_binned"] = 0.70                          # 6/7 折赢
    rows = [{"regret": {"hgb_binned1": 0.01 * i, "tclsc1": 0.02 + 0.01 * i}} for i in range(12)]
    evo = {"spearman_plain": 0.20, "spearman_lcb": 0.35, "bias_plain": 0.9, "bias_lcb": 0.3}
    g = g37.run_gates(folds, rows, evo)
    assert g["S1"]["pass"] and g["S1"]["n_folds_better"] == 6
    assert g["S2"]["pass"]
    assert g["S3"]["pass"]
    assert g37.verdict_code(g) == "SUR-WIN"
    g["S2"]["pass"] = False
    assert g37.verdict_code(g) == "SUR-RANK"
    g["S1"]["pass"] = False
    assert g37.verdict_code(g) == "SUR-DEAD"


def test_s3_requires_bias_halved():
    folds = {f: {"hgb_binned": 0.80, "tclsc": 0.72} for f in "ABCDEFG"}
    rows = [{"regret": {"hgb_binned1": 0.0, "tclsc1": 0.01}} for _ in range(8)]
    evo = {"spearman_plain": 0.20, "spearman_lcb": 0.35, "bias_plain": 0.9, "bias_lcb": 0.6}
    assert not g37.run_gates(folds, rows, evo)["S3"]["pass"]
