"""下游评估协议变体（协议锁 `results/protocol_probe/PROTOCOL.md` §3）。

**分类任务专用**（铁律 7：本模块只在 task_type=classification 的分发点被引用，
不出现在任务无关模块里）。

全部变体消费**同一份**时间维 max-pool 后的 (N, 320) 特征——表征逐位相同，唯一
变量是特征之后的分类器与它看到的训练样本量。这正是本探针要测的东西：贴顶
（AWR 零训练表征 + RBF SVM 拿到 0.990）到底是表征的属性，还是下游协议的属性。

⚠️ **协议 A 是 AutoCL 附录 B.1 的原样实现，不是"旧的那个"**。核对原文
（reference/AutoCLS.pdf p.10）："For classification, the downstream model is the SVM
classifier with RBF kernel"。B.1 里的 "linear" 属于**预测任务**（linear regression
with l2 norm）。所以 B/B′/C 是**主动偏离所复现的论文**，正当性与代价见 PROTOCOL §1。
A 因此直接委托 `linear_eval.fit_svm`，**绝不 fork**——它同时是全部历史 R' 缓存
（260 条 canonical + 640 条解耦）的口径，A 那一列复算不出旧值则整份对比无意义
（tests/test_protocol_probe.py 逐位断言）。
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .linear_eval import fit_svm

PROTOCOL_A = "A"
PROTOCOL_B = "B"
PROTOCOL_B_PRIME = "B_prime"

# few-shot 档与重复次数（PROTOCOL §3 写死）：单次 few-shot 抽样方差很大，
# 只报点估会误导，故每档 5 次抽样报均值 + std。
FEWSHOT_K = (5, 10, 20)
FEWSHOT_REPEATS = 5

PROTOCOL_NAMES = (PROTOCOL_A, PROTOCOL_B, PROTOCOL_B_PRIME) + \
    tuple(f"C_k{k}" for k in FEWSHOT_K)

# 超参网格（一次写死，跑完不调——PROTOCOL §3「不为凑结果调参」）：
# A/B′ 沿用 linear_eval 的 {1,10,100}；B 的 C 在 logistic 回归里语义不同
# （正则强度的倒数，sklearn 默认 1.0），故以默认值为中心跨 5 个数量级。
SVM_C_GRID = [1.0, 10.0, 100.0]
LOGREG_C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2]
CV_FOLDS = 3          # 与 fit_svm 对齐


def fewshot_indices(y: np.ndarray, *, k: int, seed) -> np.ndarray:
    """每类抽 ≤k 个训练样本的索引（升序）。

    类样本数不足 k 时取该类**全部**——不得丢类：训练集里缺席的类在 eval 侧仍会
    出现，分类器永远判不对，那测的就不是样本量效应而是标签空间残缺。
    seed 透传给 default_rng（可为 int 或序列，便于按 (base_seed, 重复号) 派生）。
    """
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        take = min(k, len(idx))
        out.append(rng.permutation(idx)[:take])
    return np.sort(np.concatenate(out))


def _metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {"acc": float(accuracy_score(y_true, pred)),
            "f1": float(f1_score(y_true, pred, average="macro"))}


def _grid_fit(estimator, grid: dict, X: np.ndarray, y: np.ndarray):
    gs = GridSearchCV(estimator, grid, cv=CV_FOLDS, n_jobs=-1)
    gs.fit(X, y)
    return gs


def _protocol_a(Xtr, ytr, Xev, yev, *, seed: int) -> dict:
    """现状 = AutoCL B.1：RBF SVM，C∈{1,10,100}，cv=3。直接委托 fit_svm。"""
    clf = fit_svm(Xtr, ytr, seed=seed)
    return {**_metrics(yev, clf.predict(Xev)),
            "classifier": "svm_rbf", "scaler": None, "hp": {"C": float(clf.C)}}


def _protocol_b(Xtr, ytr, Xev, yev, *, seed: int) -> dict:
    """主候选：标准化 + L2 逻辑回归（线性探针）。

    StandardScaler 放进 Pipeline 而非提前 fit——否则 CV 折内会看到验证折的统计量。
    """
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(penalty="l2", max_iter=2000))])
    gs = _grid_fit(pipe, {"clf__C": LOGREG_C_GRID}, Xtr, ytr)
    return {**_metrics(yev, gs.predict(Xev)),
            "classifier": "logreg_l2", "scaler": "standard",
            "hp": {"C": float(gs.best_params_["clf__C"])}}


def _protocol_b_prime(Xtr, ytr, Xev, yev, *, seed: int) -> dict:
    """归因对照：只把核从 RBF 换成 linear，其余（无标准化、同 C 网格）与 A 相同。"""
    gs = _grid_fit(SVC(kernel="linear"), {"C": SVM_C_GRID}, Xtr, ytr)
    return {**_metrics(yev, gs.predict(Xev)),
            "classifier": "svm_linear", "scaler": None,
            "hp": {"C": float(gs.best_params_["C"])}}


def _protocol_c(Xtr, ytr, Xev, yev, *, seed: int, k: int) -> dict:
    """few-shot probe：每类 k 个训练样本，**分类器沿用 A**（只改样本量这一个旋钮）。

    eval 侧不变——变的是"下游有多少监督"，不是"考多少题"。
    """
    accs, f1s, cs, used = [], [], [], []
    for m in range(FEWSHOT_REPEATS):
        idx = fewshot_indices(ytr, k=k, seed=[seed, m])
        clf = fit_svm(Xtr[idx], ytr[idx], seed=seed)
        met = _metrics(yev, clf.predict(Xev))
        accs.append(met["acc"])
        f1s.append(met["f1"])
        cs.append(float(clf.C))
        used.append(int(len(idx)))
    return {
        "acc": float(np.mean(accs)), "f1": float(np.mean(f1s)),
        "classifier": "svm_rbf", "scaler": None, "k": k,
        "hp": {"C_per_repeat": cs},
        "r_prime_repeats": [float(a) for a in accs],
        "r_prime_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "n_train_used": int(np.mean(used)),
    }


def evaluate_protocol(
    name: str,
    train_feats: np.ndarray,
    y_train: np.ndarray,
    eval_feats: np.ndarray,
    y_eval: np.ndarray,
    *,
    seed: int = 0,
) -> dict:
    """在**同一份特征**上跑指定下游协议 → {'r_prime','acc','f1','classifier','hp',...}。

    r_prime 语义与既有评估器一致（classification: = ACC，大者优，铁律 8 的方向仍由
    task_types.is_better 判定，本模块不做方向比较）。
    """
    if name == PROTOCOL_A:
        out = _protocol_a(train_feats, y_train, eval_feats, y_eval, seed=seed)
    elif name == PROTOCOL_B:
        out = _protocol_b(train_feats, y_train, eval_feats, y_eval, seed=seed)
    elif name == PROTOCOL_B_PRIME:
        out = _protocol_b_prime(train_feats, y_train, eval_feats, y_eval, seed=seed)
    elif name.startswith("C_k") and name in PROTOCOL_NAMES:
        out = _protocol_c(train_feats, y_train, eval_feats, y_eval,
                          seed=seed, k=int(name[3:]))
    else:
        raise ValueError(f"协议 {name!r} 未登记（PROTOCOL.md §3 写死 {PROTOCOL_NAMES}）")
    out["protocol"] = name
    out["r_prime"] = out["acc"]
    return out


def evaluate_all_protocols(
    train_feats: np.ndarray,
    y_train: np.ndarray,
    eval_feats: np.ndarray,
    y_eval: np.ndarray,
    *,
    seed: int = 0,
    names: Optional[Sequence[str]] = None,
) -> dict[str, dict]:
    """一份特征 → 全部协议变体（闸门 1/2 的离线评估入口）。"""
    return {n: evaluate_protocol(n, train_feats, y_train, eval_feats, y_eval, seed=seed)
            for n in (names or PROTOCOL_NAMES)}
