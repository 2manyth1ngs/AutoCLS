"""MX-4 WP5：学习曲线外推器模块（autocls/search/learning_curve.py）测试——先于实现（铁律 5）。

协议：results/mx4_lc_extrapolator/PROTOCOL.md §2。模块任务无关（铁律 7）：只见
观测分矩阵 / CLS 编码 / 附加常量三块向量，目标 = 增益（最终分 − 最后观测分），
predict 返回预测的最终分 = 最后观测分 + 预测增益。全部零 GPU。
"""
from __future__ import annotations

import numpy as np
import pytest

from autocls.search import learning_curve as lc


def _toy(n: int = 200, seed: int = 0):
    """增益由 CLS 第 0 位决定（+0.2 / −0.1），观测分随机；extra 是常量。"""
    rng = np.random.default_rng(seed)
    obs = rng.uniform(0.3, 0.7, size=(n, 1))
    cls_X = np.zeros((n, 3))
    cls_X[: n // 2, 0] = 1.0
    cls_X[n // 2:, 1] = 1.0
    gain = np.where(cls_X[:, 0] == 1.0, 0.2, -0.1)
    y_final = obs[:, -1] + gain
    extra = np.tile([2.5, 2.0, 5.0], (n, 1))
    return obs, cls_X, extra, y_final


# ---------------------------------------------------------------- 特征拼接
def test_curve_features_hstacks_observed_cls_extra():
    obs = np.array([[0.5, 0.6], [0.4, 0.5]])
    cls_X = np.array([[1.0, 0.0], [0.0, 1.0]])
    extra = np.array([[2.0], [3.0]])
    X = lc.curve_features(obs, cls_X=cls_X, extra=extra)
    assert X.shape == (2, 5)
    np.testing.assert_allclose(X[0], [0.5, 0.6, 1.0, 0.0, 2.0])


def test_curve_features_without_cls_or_extra_is_observed_only():
    obs = np.array([[0.5], [0.4]])
    np.testing.assert_allclose(lc.curve_features(obs), obs)


def test_curve_features_rejects_mismatched_rows_or_1d_observed():
    with pytest.raises(ValueError):
        lc.curve_features(np.array([[0.5], [0.4]]), cls_X=np.zeros((3, 2)))
    with pytest.raises(ValueError):
        lc.curve_features(np.array([0.5, 0.4]))


# ---------------------------------------------------------------- 外推器
@pytest.mark.parametrize("kind", ["ridge", "hgb"])
def test_extrapolator_learns_cls_dependent_gain(kind):
    obs, cls_X, extra, y = _toy()
    model = lc.LearningCurveExtrapolator(kind=kind, seed=0).fit(obs, y, cls_X=cls_X, extra=extra)
    q_obs = np.array([[0.5], [0.5]])
    q_cls = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    q_extra = np.tile([2.5, 2.0, 5.0], (2, 1))
    pred = model.predict(q_obs, cls_X=q_cls, extra=q_extra)
    assert pred.shape == (2,)
    assert pred[0] == pytest.approx(0.7, abs=0.03)     # 观测 0.5 + 增益 0.2
    assert pred[1] == pytest.approx(0.4, abs=0.03)     # 观测 0.5 − 0.1


def test_extrapolator_zero_gain_returns_observed_for_ridge():
    obs, cls_X, extra, _ = _toy()
    model = lc.LearningCurveExtrapolator(kind="ridge").fit(obs, obs[:, -1], cls_X=cls_X, extra=extra)
    pred = model.predict(obs[:5], cls_X=cls_X[:5], extra=extra[:5])
    np.testing.assert_allclose(pred, obs[:5, -1], atol=1e-6)


def test_extrapolator_use_cls_false_ignores_cls_matrix():
    obs, cls_X, extra, y = _toy()
    model = lc.LearningCurveExtrapolator(kind="ridge", use_cls=False).fit(obs, y, extra=extra)
    a = model.predict(obs[:5], cls_X=cls_X[:5], extra=extra[:5])
    b = model.predict(obs[:5], cls_X=1.0 - cls_X[:5], extra=extra[:5])
    np.testing.assert_allclose(a, b)


def test_extrapolator_use_cls_true_requires_cls_matrix():
    obs, cls_X, extra, y = _toy()
    with pytest.raises(ValueError):
        lc.LearningCurveExtrapolator(kind="hgb").fit(obs, y, extra=extra)


def test_extrapolator_predict_before_fit_and_shape_mismatch_fail_fast():
    obs, cls_X, extra, y = _toy()
    model = lc.LearningCurveExtrapolator(kind="ridge")
    with pytest.raises(RuntimeError):
        model.predict(obs, cls_X=cls_X, extra=extra)
    model.fit(obs, y, cls_X=cls_X, extra=extra)
    with pytest.raises(ValueError):
        model.predict(np.hstack([obs, obs])[:5], cls_X=cls_X[:5], extra=extra[:5])   # 观测截点数变了


def test_extrapolator_same_seed_is_bitwise_reproducible():
    obs, cls_X, extra, y = _toy()
    a = lc.LearningCurveExtrapolator(kind="hgb", seed=3).fit(obs, y, cls_X=cls_X, extra=extra)
    b = lc.LearningCurveExtrapolator(kind="hgb", seed=3).fit(obs, y, cls_X=cls_X, extra=extra)
    np.testing.assert_array_equal(a.predict(obs, cls_X=cls_X, extra=extra),
                                  b.predict(obs, cls_X=cls_X, extra=extra))


def test_extrapolator_rejects_unknown_kind():
    with pytest.raises(ValueError):
        lc.LearningCurveExtrapolator(kind="mlp")


# ---------------------------------------------------------------- 无参数斜率外推
def test_slope_extrapolate_two_points_is_log_linear():
    obs = np.array([[0.80, 0.85], [0.90, 0.60]])
    pred = lc.slope_extrapolate(obs, budgets=(60, 120), b_final=290)
    k = np.log(290 / 120) / np.log(120 / 60)
    np.testing.assert_allclose(pred, [0.85 + 0.05 * k, 0.60 - 0.30 * k])


def test_slope_extrapolate_single_point_returns_observed():
    obs = np.array([[0.80], [0.90]])
    np.testing.assert_allclose(lc.slope_extrapolate(obs, budgets=(60,), b_final=290), [0.80, 0.90])


def test_slope_extrapolate_validates_budgets():
    obs = np.array([[0.8, 0.9]])
    with pytest.raises(ValueError):
        lc.slope_extrapolate(obs, budgets=(120, 60), b_final=290)
    with pytest.raises(ValueError):
        lc.slope_extrapolate(obs, budgets=(60, 120), b_final=100)
    with pytest.raises(ValueError):
        lc.slope_extrapolate(obs, budgets=(60,), b_final=290)     # 列数与 budgets 不等
