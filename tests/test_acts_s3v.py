"""WP17-ACTS S3-V（results/acts/s3v/PROTOCOL.md）——先于实现。覆盖：官方 TRAIN 内分层切 D_val、验证集选择、读数码、
原始窗口 MLP 编码器、去掉 shared 的训练掩码、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _v():
    return importlib.import_module("scripts.108_acts_s3v")


def _r():
    return importlib.import_module("scripts.105_acts_s3r")


def test_whole_arrays_val():
    ac = importlib.import_module("scripts.acts_common")
    rng = np.random.default_rng(0)
    y = np.r_[np.repeat([0, 1, 2], [20, 30, 10]), np.repeat([0, 1, 2], 5)]      # 前 60 = 官方 TRAIN，后 15 = TEST
    X = rng.normal(size=(75, 16, 2)).astype(np.float32)
    a = ac.whole_arrays_val(X, y, n_train=60, val_frac=0.2, seed=7)
    tr, va = a["train_idx"], a["eval_idx"]
    assert len(set(tr) & set(va)) == 0 and sorted(set(tr) | set(va)) == list(range(60))    # 只在官方 TRAIN 内切
    assert len(va) == 12 and all((y[va] == k).sum() >= 1 for k in (0, 1, 2))                 # 分层
    b = ac.whole_arrays_val(X, y, n_train=60, val_frac=0.2, seed=7)
    assert np.array_equal(a["eval_idx"], b["eval_idx"])                                     # 固定种子
    assert np.allclose(a["Xtr"].reshape(-1, 2).mean(0), 0, atol=1e-5)                       # D_train 统计量标准化
    assert a["Xte_train"].shape == (12, 16, 2) and np.array_equal(a["Xte_train"], a["Xte_split"])


def test_select_by_val():
    v = _v()
    assert v.select_by_val(["a", "b", "c"], {"a": 0.7, "b": 0.8, "c": 0.8}) == "b"          # 平局取排名靠前
    assert v.select_by_val(["a", "b", "c"], {"a": 0.9, "b": 0.8}) == "a"                    # 缺格跳过


def test_verdict():
    v = _v()
    abl = {"w/o SetTF": 0.63, "w/o TS2Vec": 0.62, "w/o shared": 0.61}
    assert v.verdict_s3v(0.650, abl) == "S3V-BEATS-BOTH+ABL"
    assert v.verdict_s3v(0.630, abl) == "S3V-BEATS-TS2VEC"
    assert v.verdict_s3v(0.620, abl) == "S3V-BELOW"
    assert v.verdict_s3v(0.650, {**abl, "w/o shared": 0.66}) == "S3V-BEATS-BOTH"


def test_raw_mlp_encoder():
    import torch
    v = _v()
    enc = v.RawMLPSetEncoder(nz=32, hidden=8)
    E = enc.set_encode([torch.randn(5, 20, 3), torch.randn(4, 12, 1)])                     # 通道数、长度可不同
    assert E.shape == (2, 32) and enc.nz == 32


def test_train_use_mask():
    r = _r()
    m = r.train_use_mask(50, np.array([33, 40]), no_shared=True)
    assert not m[:30].any() and m[30:].sum() == 18 and not m[33] and not m[40]
    m2 = r.train_use_mask(50, np.array([33]), no_shared=False)
    assert m2[:30].all() and m2.sum() == 49


def test_cli():
    a = _v().build_parser().parse_args(["--search", "--arm", "FULL", "--dataset", "Cricket"])
    assert a.search and a.arm == "FULL" and a.dataset == "Cricket"
    b = _r().build_parser().parse_args(["--train", "--e-mode", "raw", "--no-shared"])
    assert b.e_mode == "raw" and b.no_shared
