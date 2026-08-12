# -*- coding: utf-8 -*-
"""结构探针第二轮（细化）：
1. Epilepsy：文件按类排序（4 类块）→ 在**类块内**做 k=3 聚类查参与者连续性；
   并定位第一轮发现的近重复对（cosine>0.95）。
2. FaceDetection：对 589 等块假设做定点验证——每个候选边界 589k 的跨界特征距离
   vs 块内同距离对照（z 分数），比第一轮的变点扫描更直接。
"""
from __future__ import annotations

import json
import sys
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
FD_TRAIN_NPZ = OUT / "facedetection_train_cache.npz"


def per_case_features(X: np.ndarray) -> np.ndarray:
    std = np.log(X.std(axis=1) + 1e-8)
    mean = X.mean(axis=1)
    f = np.concatenate([std, mean], axis=1)
    return (f - f.mean(axis=0)) / (f.std(axis=0) + 1e-8)


def _runs(labels: np.ndarray) -> list[list[int]]:
    runs: list[list[int]] = []
    cur, cnt = int(labels[0]), 1
    for v in labels[1:]:
        if int(v) == cur:
            cnt += 1
        else:
            runs.append([cur, cnt])
            cur, cnt = int(v), 1
    runs.append([cur, cnt])
    return runs


def probe_epilepsy() -> dict:
    from sklearn.cluster import KMeans

    X, y = load_uea_cached("Epilepsy", "TRAIN")
    res: dict = {"per_class_blocks": []}

    # 类块边界（文件按类排序，第一轮 transitions=3）
    bounds = [0] + [i + 1 for i in range(len(y) - 1) if y[i + 1] != y[i]] + [len(y)]
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        feat = per_case_features(X[b0:b1])
        km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(feat)
        runs = _runs(km.labels_)
        res["per_class_blocks"].append({
            "cls": int(y[b0]), "n": int(b1 - b0),
            "n_runs": len(runs), "runs": runs,
            "cluster_sizes": np.bincount(km.labels_).tolist(),
        })

    # 近重复对定位（全 137 展平余弦）
    F = X.reshape(len(X), -1).astype(np.float64)
    F = F - F.mean(axis=1, keepdims=True)
    F /= (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    G = F @ F.T
    np.fill_diagonal(G, -2.0)
    iu = np.triu_indices(len(X), k=1)
    hot = [(int(i), int(j), float(G[i, j])) for i, j in zip(*iu) if G[i, j] > 0.95]
    res["near_duplicate_pairs"] = [
        {"i": i, "j": j, "cosine": round(c, 6), "same_class": bool(y[i] == y[j])}
        for i, j, c in hot
    ]
    return res


def probe_fd_boundaries() -> dict:
    d = np.load(FD_TRAIN_NPZ)
    X = d["X"]
    feat = per_case_features(X)

    win = 60

    def block_dist(pos: int) -> float:
        a = feat[pos - win:pos].mean(axis=0)
        b = feat[pos:pos + win].mean(axis=0)
        return float(1.0 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    out = {"win": win, "boundaries": []}
    rng = np.random.default_rng(0)
    for k in range(1, 10):
        pos = 589 * k
        d_cross = block_dist(pos)
        # 对照：同块内部随机位置（避开边界 ±120）的同距离
        ctrl = []
        for _ in range(40):
            b = int(rng.integers(0, 10))
            lo, hi = b * 589 + win + 120, (b + 1) * 589 - win - 120
            if hi <= lo:
                continue
            p = int(rng.integers(lo, hi))
            if min(abs(p - 589 * m) for m in range(1, 10)) < 120:
                continue
            ctrl.append(block_dist(p))
        mu, sd = float(np.mean(ctrl)), float(np.std(ctrl) + 1e-12)
        out["boundaries"].append({
            "k": k, "pos": pos, "cross_dist": round(d_cross, 5),
            "ctrl_mean": round(mu, 5), "ctrl_std": round(sd, 5),
            "zscore": round((d_cross - mu) / sd, 2),
        })
    return out


def main() -> None:
    res_e = probe_epilepsy()
    res_f = probe_fd_boundaries()
    (OUT / "structure2_epilepsy.json").write_text(
        json.dumps(res_e, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "structure2_facedetection.json").write_text(
        json.dumps(res_f, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[epilepsy] near-dup:", res_e["near_duplicate_pairs"])
    print("[epilepsy] per-class n_runs:", [b["n_runs"] for b in res_e["per_class_blocks"]])
    print("[fd] boundary zscores:", [b["zscore"] for b in res_f["boundaries"]])


if __name__ == "__main__":
    main()
