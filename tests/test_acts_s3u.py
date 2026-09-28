"""WP17-ACTS S3-U（results/acts/s3u/PROTOCOL.md）——先于实现。覆盖：源子集抽样、曲线读数码、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _g():
    return importlib.import_module("scripts.107_acts_s3u")


def test_subsets():
    g = _g()
    pool = [f"s{j}" for j in range(16)]
    S = g.source_subsets(pool, 4, 30, "HW")
    assert len(S) == 30 and all(len(x) == 4 and len(set(x)) == 4 and set(x) <= set(pool) for x in S)
    assert g.source_subsets(pool, 4, 30, "HW") == S                       # 固定种子
    assert g.source_subsets(pool, 4, 30, "PS") != S                       # 按折派生
    assert g.source_subsets(pool, 16, 30, "HW") == [sorted(pool)]         # 全集只一个


def test_verdict():
    g = _g()
    assert g.verdict_s3u(g8=0.005, g12=0.007, g16=0.008, p_g=0.95, dc=0.003, p_c=0.95) == "S3U-RISING+COV"
    assert g.verdict_s3u(g8=0.005, g12=0.009, g16=0.008, p_g=0.95, dc=0.0, p_c=0.5) == "S3U-FLAT"
    assert g.verdict_s3u(g8=0.005, g12=0.006, g16=0.008, p_g=0.80, dc=0.004, p_c=0.92) == "S3U-FLAT+COV"


def test_cli():
    a = _g().build_parser().parse_args(["--analyze"])
    assert a.analyze and not a.smoke
