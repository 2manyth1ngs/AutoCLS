"""WP17-ACTS S3-R（results/acts/s3r/PROTOCOL.md）——先于实现。覆盖：验证留出、配对清洗与权重、按任务分组的批次、加权准确率、读数码、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _g():
    return importlib.import_module("scripts.105_acts_s3r")


def test_val_holdout():
    g = _g()
    h = g.val_holdout(20, "Cricket_a00")
    assert len(h) == 5 and all(30 <= x < 50 for x in h) and list(h) == sorted(h)
    assert list(g.val_holdout(20, "Cricket_a00")) == list(h)            # 固定种子
    assert len(g.val_holdout(10, "X_a01")) == 2 and len(g.val_holdout(3, "X_a02")) == 0


def test_pair_table():
    g = _g()
    r = np.array([0.9, 0.8, 0.5, 0.4])                                  # 后两个失败
    fail = np.array([False, False, True, True])
    ns = np.array([2, 2, 2, 2]); use = np.array([True, True, True, True])
    a, b, y, w, kind = g.pair_table(r, fail, 0.01, ns, use)
    pairs = set(zip(a.tolist(), b.tolist()))
    assert (2, 3) not in pairs and (3, 2) not in pairs                  # 失败对失败剔除
    assert (0, 1) in pairs and (1, 0) in pairs                          # 两个方向都放
    k = {(int(x), int(z)): (int(yy), float(ww), int(kk)) for x, z, yy, ww, kk in zip(a, b, y, w, kind)}
    assert k[(0, 1)][0] == 1 and k[(1, 0)][0] == 0 and k[(0, 1)][2] == 0
    assert k[(2, 0)] == (0, 0.25, 1) and k[(0, 2)] == (1, 0.25, 1)      # 失败对可用：可用者胜、权重 0.25
    assert len(pairs) == 2 + 8


def test_pair_table_mask_and_tie():
    g = _g()
    r = np.array([0.9, 0.9, 0.7]); fail = np.zeros(3, bool); ns = np.array([2, 2, 1])
    a, b, y, w, kind = g.pair_table(r, fail, 0.01, ns, np.array([True, True, False]))
    assert set(zip(a.tolist(), b.tolist())) == set()                    # 平局权重 0 不进；留出配置不进训练


def test_val_pairs():
    g = _g()
    r = np.array([0.9, 0.8, 0.7, 0.6]); fail = np.zeros(4, bool); ns = np.array([2, 2, 1, 1])
    a, b, *_ = g.val_pairs(r, fail, 0.001, ns, np.array([3]))
    assert all(3 in (x, z) for x, z in zip(a, b)) and len(a) == 6


def test_batches():
    g = _g()
    by_task = {0: 130, 1: 10, 2: 64, 3: 70, 4: 1}
    B = g.task_batches(by_task, np.random.default_rng(0), per_task=64, tasks_per_batch=4)
    seen = {t: 0 for t in by_task}
    for batch in B:
        assert len({t for t, _ in batch}) <= 4
        for t, idx in batch:
            assert len(idx) <= 64; seen[t] += len(idx)
    assert seen == by_task


def test_weighted_acc():
    g = _g()
    assert abs(g.weighted_acc(np.array([0.9, 0.2, 0.6]), np.array([1, 0, 0]), np.array([1.0, 1.0, 2.0])) - 0.5) < 1e-12


def test_verdict():
    g = _g()
    ok = {"ok": True}; no = {"ok": False}
    assert g.verdict_s3r(no, ok, ok) == "S3R-E-UNUSED"
    assert g.verdict_s3r(ok, no, no) == "S3R-E-FITS"
    assert g.verdict_s3r(ok, ok, ok) == "S3R-E-WIN+CONTENT"
    assert g.verdict_s3r(ok, no, ok) == "S3R-E-FITS+CONTENT"


def test_cli():
    a = _g().build_parser().parse_args(["--train", "--arm", "E", "--fold", "Cricket", "--seed", "1"])
    assert a.train and a.arm == "E" and a.fold == "Cricket" and a.seed == 1
