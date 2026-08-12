"""目标池两尺度预检的单测（铁律 5：先测后写）。

覆盖 scripts/6_target_preflight.py 的可测核心（协议 results/target_preflight/
PROTOCOL.md，判据先于数据 commit）：
1. 注册表口径：Epilepsy/FaceDetection 目标项带切片参数；FD 2 类 → allow_repeat；
   两者走源加载路径必须 fail-fast（铁律 6）。
2. Epilepsy 预处理：#104（#102 逐位重复）剔除的 fail-fast 语义。
3. FaceDetection 受试者块：n=5890 → 10 组 × 589；规模不符 fail-fast。
4. 三分判定与多切分增益的纯函数。
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autocls.data.source_pool import (  # noqa: E402
    ROLE_TARGET, TargetPoolLeakageError, build_slice_strategy, load_registry,
    load_source_dataset,
)

_pf = importlib.import_module("scripts.6_target_preflight")


# ---------------------------------------------------------------------------
# 注册表口径（铁律 6 + 切片唯一真源）
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry():
    return load_registry()


@pytest.mark.parametrize("name,n_channels,n_classes,l_enc,ncr", [
    ("Epilepsy", 3, 4, 206, (3, 4)),
    ("FaceDetection", 144, 2, 62, (2, 2)),
])
def test_target_entries_have_slice_params(registry, name, n_channels, n_classes,
                                          l_enc, ncr):
    e = registry.get(name)
    assert e.role == ROLE_TARGET
    assert (e.n_channels, e.n_classes, e.l_enc) == (n_channels, n_classes, l_enc)
    assert tuple(e.n_classes_range) == ncr
    assert e.planned_tasks == 8
    assert e.crop_len is None


def test_fd_slicer_allow_repeat(registry):
    """2 类 → 唯一 (2,2) 组合 < 8 任务，多样性须由通道/样本子集承担。"""
    s = build_slice_strategy(registry.get("FaceDetection"))
    assert s.allow_repeat
    # Epilepsy [3,4] 档组合数 C(4,3)+C(4,4)=5 < 8 → 同样 allow_repeat
    assert build_slice_strategy(registry.get("Epilepsy")).allow_repeat


@pytest.mark.parametrize("name", ["Epilepsy", "FaceDetection"])
def test_source_load_fail_fast(registry, name):
    with pytest.raises(TargetPoolLeakageError):
        load_source_dataset(name, registry)


# ---------------------------------------------------------------------------
# Epilepsy 剔重（协议 §3：#104 = #102 逐位重复，剔前必须验证）
# ---------------------------------------------------------------------------


def _fake_epilepsy(n=137, dup=True):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 206, 3)).astype(np.float32)
    if dup:
        X[104] = X[102]
    y = rng.integers(0, 4, size=n).astype(np.int64)
    return X, y


def test_epilepsy_dedup_drops_registered_duplicate():
    X, y = _fake_epilepsy()
    X2, y2, info = _pf.dedup_epilepsy(X, y)
    assert X2.shape[0] == y2.shape[0] == 136
    assert info["dropped_index"] == 104
    assert info["cosine"] > 0.999
    # 剩余数据顺序保持（#104 之前逐位不动，之后前移一位）
    assert np.array_equal(X2[104], X[105])
    assert np.array_equal(X2[103], X[103])


def test_epilepsy_dedup_fail_fast_when_not_duplicate():
    """文件顺序若变化（#102/#104 不再是重复对），必须拒绝剔除而非误伤。"""
    X, y = _fake_epilepsy(dup=False)
    with pytest.raises(ValueError, match="重复对校验失败"):
        _pf.dedup_epilepsy(X, y)


# ---------------------------------------------------------------------------
# FaceDetection 受试者块
# ---------------------------------------------------------------------------


def test_fd_groups_shape():
    g = _pf.fd_subject_groups(5890)
    assert g.shape == (5890,)
    assert g.min() == 0 and g.max() == 9
    assert np.all(np.bincount(g) == 589)
    # 块内连续（组切分的前提：块 = 文件序连续段）
    assert np.all(np.diff(g) >= 0)


@pytest.mark.parametrize("n", [5889, 5891, 3524])
def test_fd_groups_fail_fast_on_wrong_size(n):
    with pytest.raises(ValueError):
        _pf.fd_subject_groups(n)


def test_fd_val_blocks_preregistered():
    """协议 §3 预注册留出块 {8,9}；对应索引区间 [4712, 5890)。"""
    fit_idx, val_idx = _pf.fd_val_split(5890)
    assert set(np.unique(_pf.fd_subject_groups(5890)[val_idx])) == {8, 9}
    assert len(val_idx) == 2 * 589
    assert len(fit_idx) == 8 * 589
    assert np.intersect1d(fit_idx, val_idx).size == 0


# ---------------------------------------------------------------------------
# 三分判定 + 多切分增益（纯函数）
# ---------------------------------------------------------------------------


def test_preflight_verdict_matrix():
    v = _pf.preflight_verdict
    assert v(True, 0.05) == "evaluable"
    assert v(True, 0.031) == "evaluable"
    assert v(True, 0.01) == "borderline"
    assert v(False, 0.05) == "borderline"
    assert v(False, 0.0) == "unevaluable"
    assert v(False, -0.02) == "unevaluable"
    # 边界恰好 0.03 算过（判据 ≥）
    assert v(False, 0.03) == "borderline"


def test_mean_split_gain_pairs_by_split():
    """按切分配对相减再平均（不是先各自平均再相减——数值相同但逐切分记录须保留）。"""
    rand = [0.90, 0.92, 0.88]
    arm = [0.93, 0.94, 0.92]
    g = _pf.mean_split_gain(rand, arm)
    assert g == pytest.approx(np.mean([0.03, 0.02, 0.04]))
    with pytest.raises(ValueError):
        _pf.mean_split_gain([0.9], [0.9, 0.91])
