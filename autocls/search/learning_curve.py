"""学习曲线外推器（MX-4 WP5）：从早截点观测分预测同一轨迹的最终分。

定位：逐次减半（SH）在 @60 / @120 淘汰时只看"当前截点分"，而先验 top-6 内 @60 与 @290 的
秩相关中位只有 0.36（results/mx4_lc_extrapolator/PROTOCOL.md §0）。本模块用源池已跑完的
轨迹学一个"观测分 ⊕ CLS 编码 ⊕ 附加常量 → 增益"的回归器，淘汰时按预测的最终分留人。

三件纯函数级的事：
1. ``curve_features``：把观测分矩阵 (n, m)、CLS 编码 (n, d)、附加常量 (n, e) 横向拼成 X；
2. ``LearningCurveExtrapolator``：目标 = 最终分 − 最后观测分（增益），回归器复用
   ``FieldSurrogate``（ridge / hgb）；``predict`` 返回 最后观测分 + 预测增益；
3. ``slope_extrapolate``：无参数的对数线性外推（≥2 个截点），作对照。

任务无关模块（铁律 7）：只见向量，不含任务类型 / 数据集概念；"观测分"用原始 R'
而非 within-task z（部署时 @60 只训了 k0 条，算不出目录口径的 z）——归一化由脚本层决定。
"""
from __future__ import annotations

import numpy as np

from .surrogate import FieldSurrogate, KINDS


def curve_features(observed, cls_X=None, extra=None) -> np.ndarray:
    """(n, m) 观测分 ⊕ (n, d) CLS 编码 ⊕ (n, e) 附加常量 → (n, m+d+e)。cls_X / extra 可省。"""
    obs = np.asarray(observed, dtype=float)
    if obs.ndim != 2 or obs.shape[1] < 1:
        raise ValueError(f"observed 应为 (n, m≥1)，得到 {obs.shape}")
    blocks = [obs]
    for name, blk in (("cls_X", cls_X), ("extra", extra)):
        if blk is None:
            continue
        arr = np.asarray(blk, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != obs.shape[0]:
            raise ValueError(f"{name} 应为 (n={obs.shape[0]}, ·)，得到 {arr.shape}")
        blocks.append(arr)
    return np.hstack(blocks)


class LearningCurveExtrapolator:
    """kind ∈ {"ridge", "hgb"}（``FieldSurrogate`` 同参）；use_cls=False 时忽略 cls_X（消融"回归均值"）。

    fit(observed, y_final, cls_X=, extra=)：目标 = y_final − observed[:, -1]；
    predict(observed, cls_X=, extra=)：返回 observed[:, -1] + 预测增益。特征布局在 fit 时记住，
    predict 时形状不一致 fail-fast。同 seed 同数据逐位复现。
    """

    def __init__(self, kind: str = "hgb", *, use_cls: bool = True, seed: int = 0, **params):
        if kind not in KINDS:
            raise ValueError(f"未知外推器类型 {kind!r}（合法：{KINDS}）")
        self.kind = kind
        self.use_cls = bool(use_cls)
        self.seed = int(seed)
        self.params = dict(params)
        self._reg: FieldSurrogate | None = None
        self._n_features: int | None = None

    def _X(self, observed, cls_X, extra) -> np.ndarray:
        if self.use_cls:
            if cls_X is None:
                raise ValueError("use_cls=True 需要 cls_X；消融请用 use_cls=False")
        else:
            cls_X = None
        return curve_features(observed, cls_X=cls_X, extra=extra)

    def fit(self, observed, y_final, *, cls_X=None, extra=None) -> "LearningCurveExtrapolator":
        X = self._X(observed, cls_X, extra)
        y = np.asarray(y_final, dtype=float)
        if y.shape != (X.shape[0],):
            raise ValueError(f"y_final 应为 (n={X.shape[0]},)，得到 {y.shape}")
        last = np.asarray(observed, dtype=float)[:, -1]
        self._reg = FieldSurrogate(kind=self.kind, seed=self.seed, **self.params).fit(X, y - last)
        self._n_features = int(X.shape[1])
        return self

    def predict(self, observed, *, cls_X=None, extra=None) -> np.ndarray:
        if self._reg is None:
            raise RuntimeError("先 fit 再 predict")
        X = self._X(observed, cls_X, extra)
        if X.shape[1] != self._n_features:
            raise ValueError(f"特征维度 {X.shape[1]} 与 fit 时 {self._n_features} 不一致（观测截点数 / 编码变了？）")
        gain, _ = self._reg.predict(X)
        return np.asarray(observed, dtype=float)[:, -1] + gain


def slope_extrapolate(observed, budgets, b_final: int) -> np.ndarray:
    """对数线性外推：y(b_final) = y(b_m) + (y(b_m) − y(b_{m−1}))·ln(b_final/b_m)/ln(b_m/b_{m−1})；m=1 时返回观测分。"""
    obs = np.asarray(observed, dtype=float)
    b = [int(x) for x in budgets]
    if obs.ndim != 2 or obs.shape[1] != len(b):
        raise ValueError(f"observed 列数 {obs.shape} 与 budgets {b} 不等")
    if b != sorted(b) or len(set(b)) != len(b):
        raise ValueError(f"budgets 必须严格升序，得到 {b}")
    if int(b_final) <= b[-1]:
        raise ValueError(f"b_final {b_final} 必须大于最后观测截点 {b[-1]}")
    if len(b) == 1:
        return obs[:, -1].copy()
    k = np.log(int(b_final) / b[-1]) / np.log(b[-1] / b[-2])
    return obs[:, -1] + (obs[:, -1] - obs[:, -2]) * k
