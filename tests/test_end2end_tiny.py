"""端到端 tiny 测试（TECHNICAL_DESIGN.md 测试清单）：小子集上完整走
增强→预训练→下游评估 流水线，验证 acc 显著高于随机水平（1/6）。

用真实 HAR 子集（har.npz 缺失时跳过）；控制在 1-2 分钟内。
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from autocls.contrastive.cl_trainer import pretrain_encoder
from autocls.data.datasets import load_har
from autocls.downstream.linear_eval import evaluate_classification
from autocls.encoders.dilated_cnn import DilatedCNNEncoder

HAR_NPZ = Path(__file__).resolve().parents[1] / "data" / "HAR" / "har.npz"

pytestmark = pytest.mark.skipif(not HAR_NPZ.exists(), reason="先运行 scripts/1_prepare_data.py")


def _subsample(X, y, n, seed=0):
    idx = np.random.default_rng(seed).permutation(X.shape[0])[:n]
    return X[idx], y[idx]


def test_end2end_tiny_har(make_cls):
    X_train, y_train, X_test, y_test = load_har(HAR_NPZ)
    X_train, y_train = _subsample(X_train, y_train, 600, seed=0)
    X_test, y_test = _subsample(X_test, y_test, 400, seed=1)

    torch.manual_seed(0)
    encoder = DilatedCNNEncoder(input_dims=X_train.shape[-1])
    losses = pretrain_encoder(  # 30 iters = 原 3 epoch（n=600, batch=64 → 10 步/遍）
        encoder, make_cls(), X_train, max_iters=30, batch_size=64, seed=0
    )
    assert all(np.isfinite(l) for l in losses)

    res = evaluate_classification(encoder, X_train, y_train, X_test, y_test)
    # 6 类随机水平 ≈ 0.167；30 iters 小子集也应显著高于随机
    assert res["acc"] > 0.5, f"端到端 acc 异常低: {res}"
