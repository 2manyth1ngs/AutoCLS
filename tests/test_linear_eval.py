"""下游线性评估测试（AutoCL B.1：分类下游 = SVM-RBF，指标 ACC/F1）。"""
import numpy as np
import pytest
import torch
from torch import nn

from autocls.downstream.linear_eval import encode_features, evaluate_classification


class _IdentityEncoder(nn.Module):
    """把输入原样作为逐时间步表示（测试用假 encoder）。"""

    def forward(self, x):
        return x


def _separable_data(n_per_class=40, T=16, C=4, seed=0):
    rng = np.random.default_rng(seed)
    X0 = rng.normal(0.0, 0.3, size=(n_per_class, T, C)).astype(np.float32)
    X1 = rng.normal(3.0, 0.3, size=(n_per_class, T, C)).astype(np.float32)
    X = np.concatenate([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


class TestEncodeFeatures:
    def test_max_pool_over_time(self):
        # TS2Vec 分类协议：实例级表示 = 逐时间步表示的时间维 max pooling
        enc = _IdentityEncoder()
        X = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
        feats = encode_features(enc, X, device="cpu")
        assert feats.shape == (2, 2)
        assert np.allclose(feats, X.max(axis=1))

    def test_batching_consistent(self):
        enc = _IdentityEncoder()
        X = np.random.default_rng(0).normal(size=(10, 8, 3)).astype(np.float32)
        a = encode_features(enc, X, device="cpu", batch_size=3)
        b = encode_features(enc, X, device="cpu", batch_size=100)
        assert np.allclose(a, b)


class TestEvaluateClassification:
    def test_separable_data_high_acc(self):
        enc = _IdentityEncoder()
        X_train, y_train = _separable_data(seed=0)
        X_test, y_test = _separable_data(seed=1)
        res = evaluate_classification(enc, X_train, y_train, X_test, y_test, device="cpu")
        assert set(res) >= {"acc", "f1"}
        assert res["acc"] > 0.95 and res["f1"] > 0.95
