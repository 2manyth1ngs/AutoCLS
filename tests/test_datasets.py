"""数据加载与预处理测试（M1，TECHNICAL_DESIGN.md 模块规格 + AutoCL B.1 协议）。

HAR：标准 UCI HAR（train 7352 / test 2947，9 通道 Inertial Signals × 128 步，6 类），
与 AutoCL/TS2Vec 的评估协议一致。原始数据不在时跳过相关用例。
Epilepsy：UCI Epileptic Seizure Recognition CSV，按 TS-TCC 协议二值化（label==1 为发作）。
"""
from pathlib import Path

import numpy as np
import pytest

from autocls.data.datasets import load_epilepsy_csv, load_har, prepare_har

RAW_HAR = Path(__file__).resolve().parents[1] / "data" / "raw" / "uci_har" / "inner" / "UCI HAR Dataset"

needs_raw_har = pytest.mark.skipif(not RAW_HAR.exists(), reason="UCI HAR 原始数据未下载")


@needs_raw_har
def test_prepare_and_load_har(tmp_path):
    npz = tmp_path / "har.npz"
    prepare_har(RAW_HAR, npz)
    X_train, y_train, X_test, y_test = load_har(npz)

    assert X_train.shape == (7352, 128, 9)
    assert X_test.shape == (2947, 128, 9)
    assert X_train.dtype == np.float32 and X_test.dtype == np.float32
    assert y_train.shape == (7352,) and y_test.shape == (2947,)
    # 标签 0..5（原始 1..6 减一）
    assert set(np.unique(y_train)) == set(range(6))
    assert set(np.unique(y_test)) == set(range(6))
    # 按 train 统计量做的按通道 z-score：train 每通道均值≈0、方差≈1
    flat = X_train.reshape(-1, 9)
    assert np.allclose(flat.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(flat.std(axis=0), 1.0, atol=1e-3)
    assert not np.isnan(X_train).any() and not np.isnan(X_test).any()


def _tiny_epilepsy_csv(path: Path, n_per_class: int = 10) -> None:
    rng = np.random.default_rng(0)
    header = "id," + ",".join(f"X{i+1}" for i in range(178)) + ",y"
    rows = [header]
    for label in range(1, 6):
        for i in range(n_per_class):
            vals = rng.normal(size=178)
            rows.append(f"R{label}.{i}," + ",".join(f"{v:.4f}" for v in vals) + f",{label}")
    path.write_text("\n".join(rows), encoding="utf-8")


def test_load_epilepsy_csv_binarized(tmp_path):
    csv = tmp_path / "epilepsy.csv"
    _tiny_epilepsy_csv(csv)
    X_train, y_train, X_test, y_test = load_epilepsy_csv(csv, test_ratio=0.2, seed=0)

    n = 50
    assert X_train.shape == (40, 178, 1) and X_test.shape == (10, 178, 1)
    assert X_train.dtype == np.float32
    # TS-TCC 协议：label==1（癫痫发作）为正类，其余为负类
    assert set(np.unique(np.concatenate([y_train, y_test]))) == {0, 1}
    # 二值化后正类占 1/5，分层切分保持比例
    assert y_train.sum() == 8 and y_test.sum() == 2
    assert X_train.shape[0] + X_test.shape[0] == n


def test_load_epilepsy_split_deterministic(tmp_path):
    csv = tmp_path / "epilepsy.csv"
    _tiny_epilepsy_csv(csv)
    a = load_epilepsy_csv(csv, test_ratio=0.2, seed=42)
    b = load_epilepsy_csv(csv, test_ratio=0.2, seed=42)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
