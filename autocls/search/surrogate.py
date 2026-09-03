"""字段结构化 CLS 代理（MX-4 WP2）：在 CLS 字段编码上拟合调用方给定的质量分数。

定位：替换 T-CLSC 做"零样本"这一段的排序器。MX-3-PQ 复算（2026-09-03）显示
T-CLSC 在 LODO 配对准确率 0.72、对目录外候选系统性高估；一个在 one-hot 字段上的
GBDT 同缓存零 GPU 得 0.79 且 regret 更低——排序器的瓶颈是"对 19 字段空间的泛化"，
不是任务嵌入。本模块只做三件纯函数级的事：

1. ``FieldSurrogate``：ridge / HistGradientBoosting 回归 + bootstrap bagging 给不确定性；
   目标 ``y`` 由调用方提供（脚本层做 within-(task, iters) 归一化），模块不知道任务。
2. ``pairwise_matrix``：把得分向量变成 ``autocls.search.rank`` 可直接消费的 P 矩阵
   （P[i,j] = σ((s_i − s_j)/τ)，对角 NaN，P + Pᵀ = 1）——symmetrize / wins /
   borda / evolution 一行不改即可换心。
3. ``field_hamming`` / ``in_hull``：到训练集最近点的"字段汉明距离"（有几个字段不同），
   作信任域——代理对远离训练分布的候选高估（进化 CLS 预测 z +0.9 vs 真 +0.05），
   搜索侧用它剪枝或做 LCB。

任务无关模块（铁律 7）：不引入任务类型模块，不含任何数据集概念；档位来自
CLSEncoder（铁律 1），本模块只见向量。
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

KINDS = ("ridge", "hgb")
# 默认超参 = 2026-09-03 复算用的那一组（LODO 配对 acc 0.788）；调用方可覆盖。
HGB_DEFAULTS = dict(max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0)
RIDGE_DEFAULTS = dict(alpha=1.0)


class FieldSurrogate:
    """kind ∈ {"ridge", "hgb"}；bagging=B>1 时做 B 次 bootstrap 重采样拟合，
    predict 返回 (均值, 跨成员标准差)；B=1 时标准差恒零。seed 控制 bootstrap 与
    HGB 的随机流，同 seed 同数据逐位复现。"""

    def __init__(self, kind: str = "hgb", *, bagging: int = 1, seed: int = 0, **params):
        if kind not in KINDS:
            raise ValueError(f"未知代理类型 {kind!r}（合法：{KINDS}）")
        if int(bagging) < 1:
            raise ValueError("bagging 必须 ≥ 1")
        self.kind = kind
        self.bagging = int(bagging)
        self.seed = int(seed)
        self.params = dict(params)
        self._models: list | None = None

    def _make(self, seed: int):
        if self.kind == "ridge":
            return Ridge(**{**RIDGE_DEFAULTS, **self.params})
        return HistGradientBoostingRegressor(
            random_state=seed, **{**HGB_DEFAULTS, **self.params})

    def fit(self, X, y) -> "FieldSurrogate":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or y.shape != (X.shape[0],):
            raise ValueError(f"X 应为 (n, d)、y 应为 (n,)，得到 {X.shape} / {y.shape}")
        if self.bagging == 1:
            self._models = [self._make(self.seed).fit(X, y)]
            return self
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        models = []
        for b in range(self.bagging):
            idx = rng.integers(0, n, size=n)
            models.append(self._make(self.seed + b).fit(X[idx], y[idx]))
        self._models = models
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        if self._models is None:
            raise RuntimeError("先 fit 再 predict")
        X = np.asarray(X, dtype=float)
        preds = np.stack([m.predict(X) for m in self._models])   # (B, n)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0) if self.bagging > 1 else np.zeros_like(mean)
        return mean, std


def pairwise_matrix(scores, tau: float = 1.0) -> np.ndarray:
    """得分向量 → P(i ≻ j) = σ((s_i − s_j)/τ)；对角 NaN（不自比），P + Pᵀ = 1。"""
    s = np.asarray(scores, dtype=float).reshape(-1)
    if tau <= 0:
        raise ValueError("tau 必须 > 0")
    d = (s[:, None] - s[None, :]) / float(tau)
    P = 1.0 / (1.0 + np.exp(-d))
    np.fill_diagonal(P, np.nan)
    return P


def field_hamming(X, X_train, field_slices: Mapping[str, slice]) -> np.ndarray:
    """每个查询行到训练集最近行的字段汉明距离（有几个字段的编码段不同）。"""
    X = np.asarray(X, dtype=float)
    T = np.asarray(X_train, dtype=float)
    if X.ndim != 2 or T.ndim != 2 or X.shape[1] != T.shape[1]:
        raise ValueError(f"X {X.shape} 与 X_train {T.shape} 维度不一致")
    D = np.zeros((X.shape[0], T.shape[0]), dtype=int)
    for sl in field_slices.values():
        a = X[:, sl]
        b = T[:, sl]
        D += (a[:, None, :] != b[None, :, :]).any(axis=-1)
    return D.min(axis=1)


def in_hull(X, X_train, field_slices: Mapping[str, slice], radius: int = 0) -> np.ndarray:
    """查询行是否落在训练集的"字段汉明半径 radius"信任域内。"""
    return field_hamming(X, X_train, field_slices) <= int(radius)
