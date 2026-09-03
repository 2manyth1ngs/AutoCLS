"""分类下游评估（AutoCL 附录 B.1，follow TS2Vec 协议）。

实例级表示 = encoder 逐时间步输出在时间维上的 max pooling（TS2Vec 分类协议）；
下游分类器 = SVM(RBF)，指标 ACC 与 macro F1。
C 用小网格 + 交叉验证选取（TS2Vec 官方实现按 10^{-4..4} 网格搜索，此处缩小网格
并在大训练集上抽样选 C 以控制耗时，最终分类器在全量训练特征上拟合）。
"""
from __future__ import annotations

from concurrent.futures.process import BrokenProcessPool
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# 进程池损坏类异常（回退串行的白名单）：路线 A 本地验证实测——被 SIGKILL 的 worker
# 重启后（= 云上 spot 抢占后的必然场景），孤儿 loky 子进程干扰新进程的
# GridSearchCV(n_jobs=-1)，报 BrokenProcessPool"failed to un-serialize"。这是环境性
# 瞬时错误而非该 (task, cls) 的属性，若任其抛出会被 evaluate_rprime 永久 NaN 进缓存
# （--resume 不重试）。loky 的异常类在 joblib 私有路径下，导入失败时退回标准库基类。
try:
    from joblib.externals.loky.process_executor import (
        BrokenProcessPool as _LokyBrokenPool,
        TerminatedWorkerError as _LokyTerminated,
    )
    _POOL_ERRORS: tuple[type[BaseException], ...] = (
        BrokenProcessPool, _LokyBrokenPool, _LokyTerminated)
except ImportError:  # pragma: no cover - joblib 内部路径变动时的兜底
    _POOL_ERRORS = (BrokenProcessPool,)


@torch.no_grad()
def encode_features(
    encoder: torch.nn.Module,
    X: np.ndarray,
    device: Optional[str] = None,
    batch_size: int = 128,
) -> np.ndarray:
    """(N, T, C) → (N, d)：逐时间步表示做时间维 max pooling。"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    was_training = encoder.training
    encoder.eval()
    feats = []
    for start in range(0, X.shape[0], batch_size):
        x = torch.from_numpy(np.ascontiguousarray(X[start : start + batch_size], dtype=np.float32))
        z = encoder(x.to(device))          # (B, T, d)
        feats.append(z.max(dim=1).values.cpu().numpy())
    if was_training:
        encoder.train()
    return np.concatenate(feats, axis=0)


SVM_GRIDS = ("legacy", "ts2vec")
# TS2Vec 官方 tasks/classification.py::fit_svm 的 C 网格（转录；官方仓库未 vendored，
# `grep SVC reference/` 为空——来源 yuezhihan/ts2vec，沿自 T-Loss 官方评估代码）
TS2VEC_C_GRID = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, np.inf]
TS2VEC_MAX_SAMPLES = 10000


def _fit_svm_ts2vec(feats: np.ndarray, y: np.ndarray) -> SVC:
    """TS2Vec 官方 fit_svm 转录（全量协议对齐 M9/M10；仅 grid="ts2vec" 时走此路）。

    官方逻辑逐行：
        nb_classes = len(unique(y)); train_size = n
        svm = SVC(C=np.inf, gamma='scale')
        if train_size // nb_classes < 5 or train_size < 50: return svm.fit(features, y)
        grid = GridSearchCV(svm, {'C': [1e-4..1e4, inf], kernel rbf, gamma scale, ...}, cv=5)
        if train_size > 10000: 分层子样本 10000（train_test_split random_state=0）
        grid.fit(features, y); return grid.best_estimator_
    官方把 SVC 其余参数（degree/coef0/tol/cache_size/max_iter=1e7 等）以单值列进网格，
    数值上等于默认值（max_iter=1e7 vs -1 无限：实践等价），此处只列真正变化的 C。
    best_estimator_ 由 GridSearchCV refit 在（子样本后的）全部特征上重拟合——与官方同。
    """
    nb_classes = int(np.unique(y).shape[0])
    n = int(feats.shape[0])
    base = SVC(C=np.inf, kernel="rbf", gamma="scale")
    if n // nb_classes < 5 or n < 50:
        return base.fit(feats, y)
    if n > TS2VEC_MAX_SAMPLES:
        from sklearn.model_selection import train_test_split
        feats, _f2, y, _y2 = train_test_split(feats, y, train_size=TS2VEC_MAX_SAMPLES,
                                              random_state=0)
    grid_params = {"C": list(TS2VEC_C_GRID)}
    grid = GridSearchCV(base, grid_params, cv=5, n_jobs=-1)
    try:
        grid.fit(feats, y)
    except _POOL_ERRORS:
        print("[fit_svm/ts2vec] 进程池损坏（BrokenProcessPool 类），串行 n_jobs=1 重试",
              flush=True)
        grid = GridSearchCV(base, grid_params, cv=5, n_jobs=1)
        grid.fit(feats, y)
    best = grid.best_estimator_
    best.cv_score_ = float(grid.best_score_)   # 记录用（可作 top-K 内的无泄漏选择依据）
    return best


def fit_svm(
    feats: np.ndarray, y: np.ndarray, seed: int = 0, search_cap: int = 3000,
    grid: str = "legacy",
) -> SVC:
    """SVM-RBF；在 ≤search_cap 的分层抽样上网格选 C，再全量拟合。

    grid（默认 "legacy"，既有路径一字未动）："ts2vec" 走官方网格转录 _fit_svm_ts2vec。
    """
    if grid not in SVM_GRIDS:
        raise ValueError(f"未知 svm grid={grid!r}（可选 {SVM_GRIDS}）")
    if grid == "ts2vec":
        return _fit_svm_ts2vec(feats, y)
    rng = np.random.default_rng(seed)
    if feats.shape[0] > search_cap:
        idx = rng.permutation(feats.shape[0])[:search_cap]
        f_sel, y_sel = feats[idx], y[idx]
    else:
        f_sel, y_sel = feats, y
    grid = GridSearchCV(
        SVC(kernel="rbf"), {"C": [1.0, 10.0, 100.0]}, cv=3, n_jobs=-1
    )
    try:
        grid.fit(f_sel, y_sel)
    except _POOL_ERRORS:
        # 进程池损坏（见 _POOL_ERRORS 注释）→ 串行重试：3C×3 折小拟合秒级，
        # 结果与并行完全一致（n_jobs 不影响取值），只慢不改值
        print("[fit_svm] 进程池损坏（BrokenProcessPool 类），串行 n_jobs=1 重试", flush=True)
        grid = GridSearchCV(
            SVC(kernel="rbf"), {"C": [1.0, 10.0, 100.0]}, cv=3, n_jobs=1
        )
        grid.fit(f_sel, y_sel)
    best = SVC(kernel="rbf", C=grid.best_params_["C"])
    best.fit(feats, y)
    return best


def evaluate_classification(
    encoder: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: Optional[str] = None,
    seed: int = 0,
    grid: str = "legacy",
) -> dict:
    """预训练后 encoder 的下游分类评估，返回 {'acc', 'f1', 'svm_c', 'svm_grid'}。

    grid 默认 "legacy"（既有口径）；"ts2vec" = 官方网格（见 fit_svm）。
    """
    train_feats = encode_features(encoder, X_train, device=device)
    test_feats = encode_features(encoder, X_test, device=device)
    clf = fit_svm(train_feats, y_train, seed=seed, grid=grid)
    pred = clf.predict(test_feats)
    return {
        "acc": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, average="macro")),
        "svm_c": float(clf.C),
        "svm_grid": grid,
    }
