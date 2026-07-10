"""HAR / Epilepsy 数据加载与预处理（TECHNICAL_DESIGN.md 模块 3 配套数据端）。

协议对齐 AutoCL（附录 B.1，follow TS2Vec）：
- HAR：标准 UCI HAR，train 7352 / test 2947，每样本 9 通道 Inertial Signals × 128 步，
  6 类活动。按 train 统计量做按通道 z-score。
- Epilepsy：UCI Epileptic Seizure Recognition（11500 × 178 单通道）。UCI 官网已下架，
  从本地 CSV 加载；按 TS-TCC（AutoCL 引文 [11]）协议把 label==1（癫痫发作）作为正类、
  其余 4 类作为负类，做分层随机切分。

数组约定：X 形状 (N, T, C) float32；y 形状 (N,) int64，标签从 0 起。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# UCI HAR 的 9 个 Inertial Signals 通道（train/test 目录下同名 _train/_test.txt）
_HAR_SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def _load_har_split(raw_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    sig_dir = raw_dir / split / "Inertial Signals"
    channels = [
        np.loadtxt(sig_dir / f"{name}_{split}.txt", dtype=np.float32)
        for name in _HAR_SIGNALS
    ]
    X = np.stack(channels, axis=-1)  # (N, 128, 9)
    y = np.loadtxt(raw_dir / split / f"y_{split}.txt", dtype=np.int64) - 1  # 1..6 -> 0..5
    return X, y


def prepare_har(raw_dir: str | Path, out_npz: str | Path) -> None:
    """从 UCI HAR 原始目录解析并缓存为 npz。z-score 统计量只用 train 计算（防泄漏）。"""
    raw_dir = Path(raw_dir)
    X_train, y_train = _load_har_split(raw_dir, "train")
    X_test, y_test = _load_har_split(raw_dir, "test")

    # total_acc 通道均值≈1g 且方差小，float32 统计量会放大 z-score 偏差 → 用 float64
    flat = X_train.reshape(-1, X_train.shape[-1]).astype(np.float64)
    mean, std = flat.mean(axis=0), flat.std(axis=0)
    X_train = ((X_train.astype(np.float64) - mean) / std)
    X_test = ((X_test.astype(np.float64) - mean) / std)

    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        X_train=X_train.astype(np.float32), y_train=y_train,
        X_test=X_test.astype(np.float32), y_test=y_test,
    )


def load_har(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} 不存在，先运行 scripts/1_prepare_data.py 下载并预处理 HAR"
        )
    d = np.load(npz_path)
    return d["X_train"], d["y_train"], d["X_test"], d["y_test"]


def load_epilepsy_csv(
    csv_path: str | Path, test_ratio: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从 Epileptic Seizure Recognition CSV 加载并分层切分。

    CSV 首列为样本 ID（字符串）、中间 178 列为时序值、末列 y∈{1..5}。
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} 不存在。UCI 官网已下架该数据集，请从 Kaggle "
            "(harunshimanto/epileptic-seizure-recognition) 下载 CSV 后放到该路径"
        )
    raw = np.genfromtxt(
        csv_path, delimiter=",", skip_header=1,
        usecols=range(1, 180), dtype=np.float64,
    )
    X = raw[:, :-1].astype(np.float32)[..., None]  # (N, 178, 1)
    y = (raw[:, -1].astype(np.int64) == 1).astype(np.int64)  # TS-TCC 协议二值化

    # 按 train 统计量 z-score（单通道）
    rng = np.random.default_rng(seed)
    # 分层切分：每类内部打乱后按比例划 test
    train_idx, test_idx = [], []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_ratio))
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])
    train_idx = np.sort(np.concatenate(train_idx))
    test_idx = np.sort(np.concatenate(test_idx))

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    mean, std = X_train.mean(), X_train.std()
    X_train = ((X_train - mean) / std).astype(np.float32)
    X_test = ((X_test - mean) / std).astype(np.float32)
    return X_train, y_train, X_test, y_test
