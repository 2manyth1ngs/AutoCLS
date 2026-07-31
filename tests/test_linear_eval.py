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


class TestFitSvmPoolFallback:
    """进程池损坏 → 串行回退（Part A 本地验证实测的失败类）。

    实测：被 SIGKILL 的 worker 重启后（= 云上 spot 抢占后必然场景），孤儿 loky
    子进程干扰新进程的 GridSearchCV(n_jobs=-1)，报 BrokenProcessPool——环境性
    瞬时错误，若不回退会被 evaluate_rprime 永久 NaN 进缓存（resume 不重试）。
    C 网格只有 3×3 折小拟合，串行重试代价秒级、结果与并行完全一致。
    """

    def test_falls_back_to_serial_on_broken_pool(self, monkeypatch):
        import autocls.downstream.linear_eval as le

        calls = []

        class _StubGrid:
            def __init__(self, est, grid, cv=None, n_jobs=None):
                self.n_jobs = n_jobs
                self.best_params_ = {"C": 10.0}

            def fit(self, X, y):
                calls.append(self.n_jobs)
                if self.n_jobs != 1:  # 并行派发即崩（模拟孤儿 loky 干扰）
                    raise le._POOL_ERRORS[0]("A task has failed to un-serialize.")
                return self

        monkeypatch.setattr(le, "GridSearchCV", _StubGrid)
        X, y = _separable_data(seed=0)
        feats = X.max(axis=1)
        clf = le.fit_svm(feats, y, seed=0)
        assert calls == [-1, 1]        # 先并行、崩溃后串行重试
        assert clf.C == 10.0           # 回退结果照常生效

    def test_non_pool_errors_still_raise(self, monkeypatch):
        # 只回退进程池损坏类：其他异常（如数据问题）必须照常抛出，不掩盖
        import autocls.downstream.linear_eval as le

        class _StubGrid:
            def __init__(self, *a, **kw):
                pass

            def fit(self, X, y):
                raise ValueError("bad data")

        monkeypatch.setattr(le, "GridSearchCV", _StubGrid)
        X, y = _separable_data(seed=0)
        with pytest.raises(ValueError, match="bad data"):
            le.fit_svm(X.max(axis=1), y, seed=0)
