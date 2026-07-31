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


def fit_svm(
    feats: np.ndarray, y: np.ndarray, seed: int = 0, search_cap: int = 3000
) -> SVC:
    """SVM-RBF；在 ≤search_cap 的分层抽样上网格选 C，再全量拟合。"""
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
) -> dict:
    """预训练后 encoder 的下游分类评估，返回 {'acc', 'f1', 'svm_c'}。"""
    train_feats = encode_features(encoder, X_train, device=device)
    test_feats = encode_features(encoder, X_test, device=device)
    clf = fit_svm(train_feats, y_train, seed=seed)
    pred = clf.predict(test_feats)
    return {
        "acc": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, average="macro")),
        "svm_c": float(clf.C),
    }
