"""WP17-ACTS S3-N（results/acts/s3n/PROTOCOL.md）——先于实现。覆盖：E_A 特征装配（训练侧标准化 / PCA）、第一道门判据、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _g():
    return importlib.import_module("scripts.101_acts_s3n")


def test_ea_features_train_side_only():
    g = _g()
    rng = np.random.default_rng(0)
    EA = rng.normal(size=(12, 25)); groups = np.array(["a"] * 4 + ["b"] * 4 + ["F"] * 4)
    Z = g.ea_block(EA, groups, "F", pca=None)
    assert np.allclose(Z[groups != "F"].mean(0), 0, atol=1e-9) and np.allclose(Z[groups != "F"].std(0), 1)
    EA2 = EA.copy(); EA2[groups == "F"] += 50.0                                        # 改留出源不影响训练侧的标准化参数
    Z2 = g.ea_block(EA2, groups, "F", pca=None)
    assert np.allclose(Z[groups != "F"], Z2[groups != "F"])
    P = g.ea_block(EA, groups, "F", pca=4)
    assert P.shape == (12, 4)


def test_gate():
    g = _g()
    tab = {("ea25", 10.0, 1): 0.025, ("eaP4", 1.0, 1): 0.01, ("sh4", 100.0, 1): -0.006}
    assert g.gate1(tab) is True
    assert g.gate1({("ea25", 10.0, 1): 0.015, ("pt4+sh4", 100.0, 1): 0.03}) is False     # 只看 E_A 族


def test_cli():
    a = _g().build_parser().parse_args(["--compute-ea", "--parts", "8", "--part", "3"])
    assert a.compute_ea and a.parts == 8 and a.part == 3
