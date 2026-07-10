"""分类下游评估（AutoCL 附录 B.1，follow TS2Vec 协议）。

实例级表示 = encoder 逐时间步输出在时间维上的 max pooling（TS2Vec 分类协议）；
下游分类器 = SVM(RBF)，指标 ACC 与 macro F1。
C 用小网格 + 交叉验证选取（TS2Vec 官方实现按 10^{-4..4} 网格搜索，此处缩小网格
并在大训练集上抽样选 C 以控制耗时，最终分类器在全量训练特征上拟合）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


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
