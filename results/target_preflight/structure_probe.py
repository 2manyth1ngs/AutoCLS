# -*- coding: utf-8 -*-
"""目标池预检前置：Epilepsy / FaceDetection 的 subject/session 结构探针（只碰 TRAIN）。

动机（UCI-HAR 教训，subject_protocol/REPORT.md）：train 内分层随机切分若让同受试者
样本跨 train/eval，随机特征近邻匹配即可抬升基线 → 测量饱和。跑预检前必须先回答
"能不能按受试者分组切分"。

已知文件头事实（.ts 头注释，本脚本做数据侧实证）：
- Epilepsy（UEA）：6 参与者，官方 train=3 人/test=3 人，**ID 已删除**。逐次表演
  （每人每活动 >=10 次），非滑窗——孪生风险低但需实证。问题：train 内 3 人的
  块结构能否从数据恢复？
- FaceDetection：官方 train=subject01-10（每人约 580-590 trial），**ID 已删除**，
  但 Kaggle 原始数据按 subject 排列——若 .ts 保留该顺序，可用变点检测恢复块边界
  → 组切分可行。5890=10x589 恰好整除，需验证是否等块。

输出：structure_epilepsy.json / structure_facedetection.json + FaceDetection TRAIN
解析缓存 npz（494MB .ts 解析约 4 min，缓存后复用；只缓存 TRAIN，铁律 6）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from autocls.data.uea import load_uea_cached  # noqa: E402

OUT = ROOT / "results" / "target_preflight"
FD_TRAIN_NPZ = OUT / "facedetection_train_cache.npz"  # TRAIN only（铁律 6：TEST 0 触碰）


def per_case_features(X: np.ndarray) -> np.ndarray:
    """每 case 的通道统计特征 (N, 2C)：log-std + mean（受试者指纹的最便宜代理）。"""
    std = np.log(X.std(axis=1) + 1e-8)   # (N, C)
    mean = X.mean(axis=1)                # (N, C)
    f = np.concatenate([std, mean], axis=1)
    f = (f - f.mean(axis=0)) / (f.std(axis=0) + 1e-8)
    return f


def contiguity(labels: np.ndarray) -> dict:
    """聚类标签沿文件序的连续性：转移次数（块结构则接近 k-1）。"""
    trans = int((labels[1:] != labels[:-1]).sum())
    return {"n_transitions": trans, "n_clusters": int(labels.max() + 1)}


def changepoint_score(feat: np.ndarray, win: int) -> np.ndarray:
    """相邻窗均值向量的余弦距离（块边界处出峰）。返回 (N-1,) 得分（无效处 0）。"""
    n = feat.shape[0]
    score = np.zeros(n - 1)
    for i in range(win, n - win):
        a = feat[i - win:i].mean(axis=0)
        b = feat[i:i + win].mean(axis=0)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-9 and nb > 1e-9:
            score[i - 1] = 1.0 - float(a @ b) / (na * nb)
    return score


def top_peaks(score: np.ndarray, k: int, min_sep: int) -> list[int]:
    """取前 k 个互相隔开 min_sep 的峰位（边界候选，返回"块起点"索引）。"""
    order = np.argsort(score)[::-1]
    peaks: list[int] = []
    for i in order:
        if all(abs(int(i) + 1 - p) >= min_sep for p in peaks):
            peaks.append(int(i) + 1)
        if len(peaks) >= k:
            break
    return sorted(peaks)


def twin_check(X: np.ndarray, max_n: int, seed: int) -> dict:
    """展平 z-score 后的两两余弦相似度：孪生（近重复）检测。"""
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    if X.shape[0] > max_n:
        idx = np.sort(rng.choice(idx, size=max_n, replace=False))
    F = X[idx].reshape(len(idx), -1).astype(np.float64)
    F = F - F.mean(axis=1, keepdims=True)
    F /= (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    G = F @ F.T
    np.fill_diagonal(G, -2.0)
    mx = G.max(axis=1)
    return {
        "n_checked": int(len(idx)),
        "max_offdiag_cosine": float(G.max()),
        "p99_nn_cosine": float(np.quantile(mx, 0.99)),
        "median_nn_cosine": float(np.median(mx)),
        "n_pairs_above_0.95": int((G > 0.95).sum() // 2),
        "n_pairs_above_0.99": int((G > 0.99).sum() // 2),
    }


def probe_epilepsy() -> dict:
    from sklearn.cluster import KMeans

    t0 = time.time()
    X, y = load_uea_cached("Epilepsy", "TRAIN")   # (137, 206, 3)
    feat = per_case_features(X)
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(feat)
    res = {
        "dataset": "Epilepsy(UEA)",
        "shape": list(X.shape),
        "class_order_transitions": int((y[1:] != y[:-1]).sum()),
        "class_counts": {int(c): int((y == c).sum()) for c in np.unique(y)},
        "kmeans3_contiguity": contiguity(km.labels_),
        "kmeans3_labels_runs": _runs(km.labels_),
        "twin": twin_check(X, max_n=137, seed=0),
        "seconds": round(time.time() - t0, 1),
    }
    return res


def _runs(labels: np.ndarray, max_runs: int = 40) -> list[list[int]]:
    """标签序列的 run-length 编码 [label, len]（截断到 max_runs 段）。"""
    runs: list[list[int]] = []
    cur, cnt = int(labels[0]), 1
    for v in labels[1:]:
        if int(v) == cur:
            cnt += 1
        else:
            runs.append([cur, cnt])
            cur, cnt = int(v), 1
    runs.append([cur, cnt])
    return runs[:max_runs]


def probe_facedetection() -> dict:
    from sklearn.cluster import KMeans

    t0 = time.time()
    if FD_TRAIN_NPZ.exists():
        d = np.load(FD_TRAIN_NPZ)
        X, y = d["X"], d["y"]
        parse_s = 0.0
    else:
        X, y = load_uea_cached("FaceDetection", "TRAIN")   # (5890, 62, 144)
        parse_s = time.time() - t0
        np.savez_compressed(FD_TRAIN_NPZ, X=X, y=y)

    feat = per_case_features(X)
    n = X.shape[0]

    # 等块假设：5890 = 10 x 589。变点检测 → 与 589 整数倍的偏差。
    score = changepoint_score(feat, win=30)
    peaks = top_peaks(score, k=9, min_sep=200)
    expected = [589 * i for i in range(1, 10)]
    peak_dev = [min(abs(p - e) for e in expected) for p in peaks] if peaks else []

    km = KMeans(n_clusters=10, n_init=10, random_state=0).fit(feat)
    # 块纯度（按等块假设 589/块）：每块多数簇占比
    purity = []
    for b in range(10):
        seg = km.labels_[b * 589:(b + 1) * 589]
        purity.append(float(np.bincount(seg).max() / len(seg)))

    res = {
        "dataset": "FaceDetection(UEA)",
        "shape": list(X.shape),
        "parse_seconds": round(parse_s, 1),
        "class_order_transitions_first600": int((y[1:600] != y[:599]).sum()),
        "class_balance_per_block_589": [
            float(y[b * 589:(b + 1) * 589].mean()) for b in range(10)
        ],
        "changepoint_peaks": peaks,
        "expected_multiples_of_589": expected,
        "peak_deviation_from_589grid": peak_dev,
        "kmeans10_contiguity": contiguity(km.labels_),
        "kmeans10_block_purity_589": purity,
        "twin": twin_check(X, max_n=2000, seed=0),
        "seconds": round(time.time() - t0, 1),
    }
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("epilepsy", probe_epilepsy), ("facedetection", probe_facedetection)):
        res = fn()
        p = OUT / f"structure_{name}.json"
        p.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[ok] {p.name}: {json.dumps(res, ensure_ascii=False)[:400]}")


if __name__ == "__main__":
    main()
