"""WP17-ACTS S3-E（results/acts/s3e/PROTOCOL.md）——先于实现。覆盖：L4 连续权重、集成胜场适应度与排序、家族分散前 3、选臂规则、读数码、臂名解析、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _g():
    return importlib.import_module("scripts.91_acts_s3e")


def test_l4_weight():
    g = _g()
    assert np.isclose(g.l4_weight(0.50, 0.50, 0.02, 2, 2), 0.0)
    s = 0.02 * np.sqrt(1 / 2 + 1 / 1)
    assert np.isclose(g.l4_weight(0.52, 0.50, 0.02, 2, 1), min(0.02 / (2 * s), 1.0))
    assert g.l4_weight(0.9, 0.1, 0.02, 1, 1) == 1.0
    assert g.l4_weight(0.9, 0.1, 0.0, 1, 1) == 1.0            # σ = 0 → 非零差视为确定


def test_ens_fitness_and_order():
    g = _g()
    W = np.array([[3, 1, 2], [3, 3, 0], [3, 2, 1]], float)    # 3 模型 × 3 候选
    f0 = g.ens_fitness(W, 0.0); f1 = g.ens_fitness(W, 1.0)
    assert np.allclose(f0, W.mean(0)) and np.allclose(f1, W.mean(0) - W.std(0))
    assert g.ens_order(W, ["c", "a", "b"], 1.0)[0] == 0        # 候选 0 均值最高且无分歧
    assert g.ens_order(np.array([[1, 1.0]]), ["z", "a"], 0.0) == [1, 0]   # 并列按 key


def test_diversified_top3():
    g = _g()
    fam = {"k1": ("a", "x", "n"), "k2": ("a", "x", "n"), "k3": ("b", "x", "n"), "k4": ("a", "y", "n"), "k5": ("c", "z", "n")}
    assert g.diversified_top(["k1", "k2", "k3", "k4", "k5"], fam, 3) == ["k1", "k3", "k4"]
    assert g.diversified_top(["k1", "k2"], fam, 3) == ["k1", "k2"]            # 不够时按原序补足


def test_select_arm():
    g = _g()
    v = {"field-off-L0": 0.010, "diff-on-L4": 0.008, "diff-off-L4": 0.008, "field-on-L0": 0.009}
    assert g.select_arm(v) == "diff-on-L4"                     # 平局：diff > field，on > off，L4 > L0
    assert g.parse_arm("field-on-L4") == ("field", True, "L4")


def test_verdict():
    g = _g()
    assert g.verdict_s3e({"S1": True, "S2": True, "S3": True}, {"R1": True, "R2": False, "R3": True}) == "S3E-SEARCH-WIN+R1+R3"
    assert g.verdict_s3e({"S1": False, "S2": True, "S3": True}, {"R1": False, "R2": False, "R3": False}) == "S3E-CATALOG-PAR"
    assert g.verdict_s3e({"S1": True, "S2": False, "S3": True}, {"R1": False, "R2": True, "R3": False}) == "S3E-SEARCH-LOSS+R2"


def test_cli():
    a = _g().build_parser().parse_args(["--train", "--arm", "diff-on-L4", "--fold", "Cricket", "--seed", "3"])
    assert a.train and a.arm == "diff-on-L4" and a.fold == "Cricket" and a.seed == 3
