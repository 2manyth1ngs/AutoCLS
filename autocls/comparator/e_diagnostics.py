"""任务嵌入 E 的特征侧退化诊断（实验 1 (a)/(b)；results/e_degradation_diag/PROTOCOL.md §1–§2）。

纯 numpy 函数，输入 (n_tasks, dim) 矩阵与分组标签；不 import 任何数据集/任务类型概念（铁律 7）。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    U = X / n
    return U @ U.T


def effective_rank(X: np.ndarray) -> float:
    """协方差谱的 participation ratio (Σλ)²/Σλ²（跨行中心化）。零协方差 → 0。"""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    # 用 Gram 谱（n×n）避免 dim×dim 大矩阵；非零特征值与协方差谱相同
    lam = np.linalg.eigvalsh(Xc @ Xc.T)
    lam = np.clip(lam, 0, None)
    s2 = float(np.sum(lam ** 2))
    if s2 <= 0:
        return 0.0
    return float(np.sum(lam) ** 2 / s2)


def relative_dispersion(X: np.ndarray) -> float:
    """mean_t‖x_t − x̄‖ / ‖x̄‖——"常数 + 微扰"的直接读数（均值为零向量时 inf）。"""
    X = np.asarray(X, dtype=np.float64)
    xbar = X.mean(axis=0)
    nb = float(np.linalg.norm(xbar))
    disp = float(np.mean(np.linalg.norm(X - xbar, axis=1)))
    if nb == 0:
        return float("inf") if disp > 0 else 0.0
    return disp / nb


def zscore_across(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """跨行逐维标准化（std 为 0 的维度置 0）。"""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=0, keepdims=True)
    Z = (X - mu) / np.where(sd < eps, 1.0, sd)
    Z[:, (sd < eps).ravel()] = 0.0
    return Z


def constancy_ratio(C: np.ndarray, c0: np.ndarray) -> float:
    """ρ_c = mean_t‖c_t − c̄‖ / ‖c̄ − c₀‖（协议 §2）。分母为零时 inf（除非分子也为零）。"""
    C = np.asarray(C, dtype=np.float64)
    cbar = C.mean(axis=0)
    num = float(np.mean(np.linalg.norm(C - cbar, axis=1)))
    den = float(np.linalg.norm(cbar - np.asarray(c0, dtype=np.float64)))
    if den == 0:
        return 0.0 if num == 0 else float("inf")
    return num / den


def group_stats(X: np.ndarray, groups: Sequence[str]) -> dict:
    """组内 / 组间平均余弦与欧氏距离、分离比（组间距 / 组内距）、余弦极值。"""
    X = np.asarray(X, dtype=np.float64)
    g = np.asarray(groups)
    C = cosine_matrix(X)
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    iu = np.triu_indices(len(X), k=1)
    same = (g[:, None] == g[None, :])[iu]
    cos_u, dist_u = C[iu], D[iu]
    within_d = float(dist_u[same].mean()) if same.any() else float("nan")
    cross_d = float(dist_u[~same].mean()) if (~same).any() else float("nan")
    return {
        "within_cos": float(cos_u[same].mean()) if same.any() else float("nan"),
        "cross_cos": float(cos_u[~same].mean()) if (~same).any() else float("nan"),
        "within_dist": within_d,
        "cross_dist": cross_d,
        "separation_ratio": (cross_d / within_d) if within_d and np.isfinite(within_d) else float("nan"),
        "min_cos": float(cos_u.min()) if len(cos_u) else float("nan"),
        "max_cos": float(cos_u.max()) if len(cos_u) else float("nan"),
        "n_within_pairs": int(same.sum()),
        "n_cross_pairs": int((~same).sum()),
    }


def layer_report(X: np.ndarray, groups: Sequence[str]) -> dict:
    """一层表征的完整读数（协议 §1）：分组统计 + 有效秩 + 相对离散度 + z-score 后分组统计。"""
    X = np.asarray(X, dtype=np.float64)
    out = group_stats(X, groups)
    out["effective_rank"] = effective_rank(X)
    out["relative_dispersion"] = relative_dispersion(X)
    out["mean_norm"] = float(np.mean(np.linalg.norm(X, axis=1)))
    var = X.var(axis=0, ddof=0)
    out["dim_var_spectrum"] = {
        "max": float(var.max()), "median": float(np.median(var)), "min": float(var.min()),
        "top5_share": float(np.sort(var)[::-1][:5].sum() / var.sum()) if var.sum() > 0 else float("nan"),
    }
    out["zscored"] = group_stats(zscore_across(X), groups)
    out["zscored"]["effective_rank"] = effective_rank(zscore_across(X))
    return out
