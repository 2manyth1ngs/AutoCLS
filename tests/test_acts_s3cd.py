"""WP17-ACTS S3-C-D（results/acts/s3cd/PROTOCOL.md）——先于实现。覆盖：任务 id、按 test acc 选 1、靶侧 ④ 特征与 S3-A probe_feature 同构、计划格去重与官方 eval、命令行。"""
from __future__ import annotations

import importlib

import numpy as np


def _g():
    return importlib.import_module("scripts.90_acts_s3cd")


def test_task_id():
    assert _g().task_id("Cricket") == "Cricket_whole_official"


def test_pick_by_test():
    g = _g()
    assert g.pick_by_test(["a", "b", "c"], {"a": 0.5, "b": 0.8, "c": 0.8}) == ("b", 0.8)


def test_target_probe_features_match_s3a():
    g = _g(); g84 = importlib.import_module("scripts.84_acts_s3a")
    rng = np.random.default_rng(0)
    acc25 = rng.uniform(0.2, 0.9, size=(1, 30)); pf = np.array([0.15]); maj = np.array([0.1])
    ctx = {"mu": rng.normal(size=30), "B": rng.normal(size=(30, 3))}
    ref = g84.probe_feature(acc25, pf, maj)(ctx)
    keys = [f"k{i}" for i in range(30)]
    P = g84.probe_configs(ctx)
    got = g.target_probe_X({keys[j]: float(acc25[0, j]) for j in P}, [keys[j] for j in P], float(pf[0]), float(maj[0]))
    assert np.allclose(got, ref[0])


def test_cli():
    a = _g().build_parser().parse_args(["--search", "--arm", "D", "--dataset", "Cricket"])
    assert a.search and a.arm == "D" and a.dataset == "Cricket"
