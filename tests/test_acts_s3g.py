"""WP17-ACTS S3-G（results/acts/s3g/PROTOCOL.md）——先于实现。覆盖：扩充任务选择、扩充标签并入、预测—真值胜率、读数码、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _g():
    return importlib.import_module("scripts.94_acts_s3g")


def test_aug_tasks():
    g = _g()
    order = ["A", "B", "C", "D"]
    tasks = [f"{s}_a{r:02d}" for s in order for r in range(8)]
    groups = [t.split("_")[0] for t in tasks]
    got = g.aug_tasks(tasks, groups, order, fold="A", val="B")
    assert got == ["C_a00", "C_a04", "D_a00", "D_a04"]


def test_merge_aug():
    g = _g()
    V = [np.zeros((3, 2)), np.ones((2, 2))]; R = [np.array([.1, .2, .3]), np.array([.5, .6])]; NS = [np.array([2, 2, 1]), np.array([2, 1])]
    aug = {1: (np.full((2, 2), 7.0), np.array([.9, .4]))}
    V2, R2, NS2 = g.merge_aug(V, R, NS, aug)
    assert V2[0] is V[0] and V2[1].shape == (4, 2) and np.allclose(R2[1], [.5, .6, .9, .4]) and NS2[1].tolist() == [2, 1, 1, 1]
    assert V[1].shape == (2, 2)                                   # 原数组不被改动


def test_beat_share():
    g = _g()
    assert np.isclose(g.true_beat(0.5, np.array([0.1, 0.5, 0.7, 0.2])), 0.5)          # 严格大于：0.1 / 0.2
    P = np.array([[0.9, 0.1], [0.7, 0.3]])                                           # 2 模型 × 2 条 shared
    assert np.isclose(g.pred_beat_from(P), 0.5)


def _c(d, p, npos):
    return {"d": d, "p": p, "npos": npos}


def test_verdict():
    g = _g()
    assert g.verdict_s3g(_c(0.012, 0.01, 5), _c(0.004, 0.2, 4), _c(0.011, 0.02, 4), _c(0.002, 0.3, 4), True) == "S3G-L3-WIN+CAT-PAR+E"
    assert g.verdict_s3g(_c(0.012, 0.01, 5), _c(-0.001, 0.2, 2), _c(0.0, 1.0, 3), _c(0.0, 1.0, 3), False) == "S3G-L3-NULL"
    assert g.verdict_s3g(_c(-0.02, 0.01, 1), _c(-0.02, 0.01, 1), _c(0.0, 1.0, 3), _c(0.0, 1.0, 3), False) == "S3G-L3-HURT"


def test_cli():
    a = _g().build_parser().parse_args(["--train-f", "--fold", "Cricket", "--seed", "4"])
    assert a.train_f and a.fold == "Cricket" and a.seed == 4
