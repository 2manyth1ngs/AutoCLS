"""UEA 多变量数据加载/预处理测试（路线 A 探针配套；铁律 5 先测后写）。

被测纯函数（不依赖 aeon，用合成数据）：
- encode_labels：字符串标签 → 0..k-1 连续 int64，保序稳定。
- pad_and_zscore：不等长序列列表 → (N, T, C) 等长，z-score 只统计真实时间步，
  右侧零填充（= z-score 后的通道均值，中性），可选 crop_len 截断长序列。

aeon 后端集成用例（缓存缺失则跳过）：load_uea_cached 返回 (N, T, C) float32 +
0..k-1 int64，通道数/长度与 UEA 调研表一致。
"""
from pathlib import Path

import numpy as np
import pytest

from autocls.data.uea import encode_labels, load_uea_cached, pad_and_zscore

CACHE = Path(__file__).resolve().parents[1] / "data" / "uea_cache"


# ---- encode_labels ----

def test_encode_labels_maps_to_contiguous_ints():
    y = np.array(["b", "a", "b", "c", "a"])
    yi, classes = encode_labels(y)
    assert yi.dtype == np.int64
    assert list(classes) == ["a", "b", "c"]          # np.unique 升序
    assert yi.tolist() == [1, 0, 1, 2, 0]
    assert set(yi.tolist()) == {0, 1, 2}


def test_encode_labels_numeric_strings():
    # UEA 标签常是数字字符串（'1'..'10'）——只要求映射为连续 int，取值即类 id
    y = np.array(["1", "10", "2", "1"])
    yi, classes = encode_labels(y)
    assert set(yi.tolist()) == {0, 1, 2}
    assert len(classes) == 3


# ---- pad_and_zscore：等长直通 ----

def test_pad_and_zscore_equal_length_passthrough_shape():
    rng = np.random.default_rng(0)
    seqs = [rng.normal(size=(10, 3)).astype(np.float64) for _ in range(5)]
    X = pad_and_zscore(seqs)
    assert X.shape == (5, 10, 3)
    assert X.dtype == np.float32


def test_pad_and_zscore_channelwise_zero_mean_unit_std():
    rng = np.random.default_rng(1)
    seqs = [rng.normal(loc=5.0, scale=3.0, size=(20, 2)).astype(np.float64) for _ in range(8)]
    X = pad_and_zscore(seqs)
    flat = X.reshape(-1, 2)
    assert np.allclose(flat.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(flat.std(axis=0), 1.0, atol=1e-3)


# ---- pad_and_zscore：不等长右侧零填充 ----

def test_pad_and_zscore_ragged_right_pads_with_zero():
    seqs = [
        np.ones((3, 2), dtype=np.float64),
        np.ones((7, 2), dtype=np.float64) * 2.0,
    ]
    X = pad_and_zscore(seqs)
    assert X.shape == (2, 7, 2)                       # 补到最长 7
    # 第 0 条真实长度 3，其后 4 步为填充零（z-score 后通道均值处 = 0）
    assert np.allclose(X[0, 3:], 0.0)
    # 填充不进入统计：整体均值仍≈0（真实步 z-score 后）
    real_rows = np.concatenate([X[0, :3], X[1, :7]], axis=0)
    assert np.allclose(real_rows.mean(axis=0), 0.0, atol=1e-6)


def test_pad_and_zscore_zscore_excludes_padding():
    # 若填充零错误地进入 mean/std 统计，短序列多的批统计量会被拉偏
    seqs = [np.full((2, 1), 4.0), np.full((2, 1), 6.0),
            np.full((2, 1), 4.0), np.full((2, 1), 6.0)]  # 全长 2，真实均值 5
    # 加一条长序列制造填充
    seqs.append(np.full((8, 1), 5.0))
    X = pad_and_zscore(seqs)
    assert X.shape == (5, 8, 1)
    # 真实步（前 4 条各 2 步 + 末条 8 步）均值应为 5 → z-score 后真实步均值 0
    real = np.concatenate([X[i, :2, 0] for i in range(4)] + [X[4, :8, 0]])
    assert abs(real.mean()) < 1e-6


# ---- pad_and_zscore：crop_len 截断 ----

def test_pad_and_zscore_crop_truncates_long_sequences():
    seqs = [np.arange(100, dtype=np.float64).reshape(100, 1),
            np.arange(30, dtype=np.float64).reshape(30, 1)]
    X = pad_and_zscore(seqs, crop_len=50)
    assert X.shape == (2, 50, 1)                      # 目标长度 = crop_len
    assert np.allclose(X[1, 30:], 0.0)               # 短序列仍右侧零填充到 50


def test_pad_and_zscore_constant_channel_no_nan():
    # 常值通道 std=0，不得除零产 NaN
    seqs = [np.concatenate([np.full((5, 1), 7.0), np.random.default_rng(0).normal(size=(5, 1))], axis=1)
            for _ in range(4)]
    X = pad_and_zscore(seqs)
    assert not np.isnan(X).any()
    assert np.allclose(X[:, :, 0], 0.0)              # 常值通道 z-score 后恒 0


# ---- aeon 后端集成（缓存缺失跳过）----

_probe_specs = [
    ("PenDigits", 2, 8, 10, False),
    ("PhonemeSpectra", 11, 217, 39, False),
    ("Cricket", 6, 1197, 12, False),
    ("SpokenArabicDigits", 13, 93, 10, True),  # 不等长，pad 到 L_max=93
]


@pytest.mark.parametrize("name,n_ch,L,n_cls,ragged", _probe_specs)
def test_load_uea_cached_shapes(name, n_ch, L, n_cls, ragged):
    if not (CACHE / name / f"{name}_TRAIN.ts").exists():
        pytest.skip(f"{name} 缓存缺失")
    X, y = load_uea_cached(name, split="TRAIN", cache_dir=CACHE)
    assert X.ndim == 3 and X.shape[2] == n_ch
    assert X.shape[1] == L                            # 等长=L；不等长=pad 到 L_max
    assert X.dtype == np.float32
    assert y.dtype == np.int64
    assert set(y.tolist()) == set(range(n_cls))       # 0..k-1 连续
    assert not np.isnan(X).any()


def test_load_uea_cached_crop_len():
    name = "Cricket"
    if not (CACHE / name / f"{name}_TRAIN.ts").exists():
        pytest.skip("Cricket 缓存缺失")
    X, y = load_uea_cached(name, split="TRAIN", cache_dir=CACHE, crop_len=256)
    assert X.shape[1] == 256                           # 长序列裁到 CROP
