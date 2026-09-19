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


# ---------------------------------------------------------------------------
# SVM 网格开关（全量协议对齐 M9/M10）：grid="legacy"（既有 C∈{1,10,100}, cv=3）
# 与 grid="ts2vec"（TS2Vec 官方 tasks/classification.py::fit_svm 转录：
# n//k<5 或 n<50 → SVC(C=inf) 直接拟合；否则 C∈{1e-4..1e4, inf}, cv=5）。
# ---------------------------------------------------------------------------


class TestSvmGrid:
    def _stub(self, monkeypatch, calls):
        import autocls.downstream.linear_eval as le

        class _StubGrid:
            def __init__(self, est, grid, cv=None, n_jobs=None):
                calls.append({"grid": grid, "cv": cv, "est": est})
                self.best_params_ = {"C": 10.0}
                self.best_estimator_ = est
                self.best_score_ = 0.5

            def fit(self, X, y):
                self.best_estimator_.fit(X, y)
                return self

        monkeypatch.setattr(le, "GridSearchCV", _StubGrid)
        return le

    def test_default_grid_is_legacy_and_unchanged(self, monkeypatch):
        import inspect
        calls = []
        le = self._stub(monkeypatch, calls)
        X, y = _separable_data(seed=0)
        le.fit_svm(X.max(axis=1), y, seed=0)
        assert calls[0]["grid"] == {"C": [1.0, 10.0, 100.0]} and calls[0]["cv"] == 3
        assert inspect.signature(le.fit_svm).parameters["grid"].default == "legacy"

    def test_ts2vec_grid_has_inf_and_cv5(self, monkeypatch):
        calls = []
        le = self._stub(monkeypatch, calls)
        X, y = _separable_data(n_per_class=40, seed=0)   # n=80, k=2 → 走网格
        clf = le.fit_svm(X.max(axis=1), y, seed=0, grid="ts2vec")
        g = calls[0]
        assert g["cv"] == 5
        cs = g["grid"]["C"]
        assert np.isinf(cs[-1]) and 1e-4 in cs and 1e4 in cs and len(cs) == 10
        assert g["est"].gamma == "scale" and g["est"].kernel == "rbf"
        # 官方 param_grid 里的 'max_iter': [10000000]：C=inf 在不可分多类数据上无界不收敛
        assert g["est"].max_iter == 10_000_000
        assert clf is g["est"]                        # 返回 best_estimator_

    def test_ts2vec_fallback_small_n_uses_c_inf(self, monkeypatch):
        calls = []
        le = self._stub(monkeypatch, calls)
        X, y = _separable_data(n_per_class=20, seed=0)   # n=40 < 50 → 回退
        clf = le.fit_svm(X.max(axis=1), y, seed=0, grid="ts2vec")
        assert calls == []                            # 未进网格
        assert np.isinf(clf.C)
        assert clf.predict(X.max(axis=1)).shape == (40,)

    def test_ts2vec_fallback_few_per_class(self, monkeypatch):
        calls = []
        le = self._stub(monkeypatch, calls)
        rng = np.random.default_rng(0)
        feats = rng.normal(size=(60, 4))
        y = np.arange(60) % 13                        # 60//13=4 < 5 → 回退
        clf = le.fit_svm(feats, y, seed=0, grid="ts2vec")
        assert calls == [] and np.isinf(clf.C)

    def test_unknown_grid_raises(self):
        from autocls.downstream.linear_eval import fit_svm
        X, y = _separable_data(seed=0)
        with pytest.raises(ValueError):
            fit_svm(X.max(axis=1), y, grid="rbf_wide")

    def test_evaluate_classification_records_grid(self):
        enc = _IdentityEncoder()
        X_train, y_train = _separable_data(seed=0)
        X_test, y_test = _separable_data(seed=1)
        res = evaluate_classification(enc, X_train, y_train, X_test, y_test, device="cpu")
        assert res["svm_grid"] == "legacy"
        res2 = evaluate_classification(enc, X_train, y_train, X_test, y_test,
                                       device="cpu", grid="ts2vec")
        assert res2["svm_grid"] == "ts2vec" and res2["acc"] > 0.95


# ---- 官方网格的 C=inf 在旧 sklearn（1.3.x，py3.8 上限）被参数校验拒绝：用有限替身（WP17-ACTS 实验室服务器） ----

def _tiny_separable(n=24, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    X = rng.standard_normal((n, 4)) + y[:, None] * 6.0
    return X.astype(np.float32), y


def test_ts2vec_svm_c_inf_kept_when_sklearn_accepts_it(monkeypatch):
    from autocls.downstream import linear_eval as le
    monkeypatch.setattr(le, "_sklearn_accepts_inf_c", lambda: True)
    X, y = _tiny_separable()
    clf = le.fit_svm(X, y, grid="ts2vec")           # n < 50 → 官方回退分支 SVC(C=inf)
    assert clf.C == np.inf


def test_ts2vec_svm_uses_finite_surrogate_when_inf_rejected(monkeypatch):
    from autocls.downstream import linear_eval as le
    monkeypatch.setattr(le, "_sklearn_accepts_inf_c", lambda: False)
    X, y = _tiny_separable()
    clf = le.fit_svm(X, y, grid="ts2vec")
    assert clf.C == le.TS2VEC_C_INF_SURROGATE and np.isfinite(clf.C) and clf.C >= 1e9
    assert (clf.predict(X) == y).mean() == 1.0
    grid = le.ts2vec_c_grid()
    assert all(np.isfinite(c) for c in grid) and grid[-1] == le.TS2VEC_C_INF_SURROGATE and len(grid) == len(le.TS2VEC_C_GRID)


def test_sklearn_accepts_inf_c_matches_installed_version():
    from autocls.downstream import linear_eval as le
    from sklearn.svm import SVC
    try:
        SVC(C=np.inf)._validate_params()
        expect = True
    except Exception:  # noqa: BLE001
        expect = False
    assert le._sklearn_accepts_inf_c() is expect

