"""候选数据集通用加载层测试（换池调研配套；铁律 5 先测后写）。

被测纯函数（合成数据，不依赖任何下载）：
- window_stream：连续传感器流 (T, C) + 逐步标签 → 定长窗 (N, win, C)。
  纯标签窗才保留（跨活动边界的混合窗丢弃）、负标签=忽略段、含 NaN 窗丢弃。
- zscore_channels：按通道 z-score（float64 统计后转 float32），常值通道不产 NaN。
- drop_rare_classes：样本数不足的类整类剔除后重新连续编号。
- describe：真实统计（N/T/C/K/类别计数/不平衡比），第1步统计表的唯一产出口径。
- save_candidate / load_candidate：canonical npz 往返，y 恒为 0..k-1 int64。

窗口口径（PROTOCOL 预注册）：win=128 步 @原生采样率、stride=64（50% 重叠）。
128 = UCI-HAR 的 L（唯一验证过"CLS 差异可见"的正样本），也满足 L≥50 的 encoder 边界。
"""
from __future__ import annotations

import numpy as np
import pytest

from autocls.data.candidates import (
    WINDOW_LEN,
    WINDOW_STRIDE,
    decimate_to_length,
    describe,
    drop_rare_classes,
    extract_archive,
    fetch_archive,
    load_candidate,
    save_candidate,
    window_stream,
    zscore_channels,
)


# ---- 预注册常量 ----

def test_window_defaults_match_uci_har_profile():
    # 128 步 = UCI-HAR 的 L；stride 64 = 50% 重叠（UCI-HAR 原始协议同）
    assert WINDOW_LEN == 128
    assert WINDOW_STRIDE == 64


# ---- window_stream：基本切窗 ----

def test_window_stream_counts_and_shapes():
    x = np.arange(300 * 2, dtype=np.float64).reshape(300, 2)
    labels = np.zeros(300, dtype=np.int64)
    X, y = window_stream(x, labels, win=100, stride=50)
    # 起点 0,50,...,200 共 5 个（250+100>300 故止于 200）
    assert X.shape == (5, 100, 2)
    assert y.shape == (5,)
    assert set(y.tolist()) == {0}


def test_window_stream_preserves_time_order_within_window():
    x = np.arange(20, dtype=np.float64).reshape(20, 1)
    labels = np.zeros(20, dtype=np.int64)
    X, _ = window_stream(x, labels, win=10, stride=10)
    np.testing.assert_allclose(X[0, :, 0], np.arange(10))
    np.testing.assert_allclose(X[1, :, 0], np.arange(10, 20))


def test_window_stream_segment_shorter_than_window_yields_nothing():
    x = np.zeros((50, 3), dtype=np.float64)
    labels = np.zeros(50, dtype=np.int64)
    X, y = window_stream(x, labels, win=128, stride=64)
    assert X.shape == (0, 128, 3)
    assert y.shape == (0,)


# ---- window_stream：混合标签窗必须丢弃 ----

def test_window_stream_drops_windows_spanning_label_change():
    x = np.zeros((40, 1), dtype=np.float64)
    labels = np.array([0] * 15 + [1] * 25, dtype=np.int64)
    X, y = window_stream(x, labels, win=10, stride=10)
    # 起点 0(全0) / 10(0,1 混合→丢) / 20(全1) / 30(全1)
    assert y.tolist() == [0, 1, 1]
    assert X.shape[0] == 3


def test_window_stream_drops_windows_touching_ignored_label():
    # 负标签 = 忽略段（Opportunity 的 null 类、PAMAP2 的 transient activity 0）
    x = np.zeros((30, 1), dtype=np.float64)
    labels = np.array([2] * 10 + [-1] * 10 + [2] * 10, dtype=np.int64)
    X, y = window_stream(x, labels, win=10, stride=5)
    # 只有完全落在两段纯 2 里的窗留下：起点 0 与 20
    assert y.tolist() == [2, 2]
    assert X.shape[0] == 2


# ---- window_stream：含 NaN 的窗丢弃（PAMAP2 心率/丢包）----

def test_window_stream_drops_windows_containing_nan():
    x = np.zeros((30, 2), dtype=np.float64)
    x[12, 1] = np.nan
    labels = np.zeros(30, dtype=np.int64)
    X, y = window_stream(x, labels, win=10, stride=10)
    assert X.shape[0] == 2                    # 起点 0 与 20 保留，10 因 NaN 丢弃
    assert not np.isnan(X).any()


def test_window_stream_rejects_mismatched_label_length():
    x = np.zeros((30, 1), dtype=np.float64)
    with pytest.raises(ValueError):
        window_stream(x, np.zeros(29, dtype=np.int64), win=10, stride=10)


# ---- zscore_channels ----

def test_zscore_channels_zero_mean_unit_std_per_channel():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=[5.0, -3.0], scale=[2.0, 0.5], size=(40, 128, 2))
    Z = zscore_channels(X)
    flat = Z.reshape(-1, 2)
    assert Z.dtype == np.float32
    assert np.allclose(flat.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(flat.std(axis=0), 1.0, atol=1e-3)


def test_zscore_channels_constant_channel_no_nan():
    X = np.concatenate(
        [np.full((10, 8, 1), 7.0), np.random.default_rng(1).normal(size=(10, 8, 1))], axis=2
    )
    Z = zscore_channels(X)
    assert not np.isnan(Z).any()
    assert np.allclose(Z[:, :, 0], 0.0)


# ---- drop_rare_classes ----

def test_drop_rare_classes_removes_and_recodes():
    X = np.zeros((10, 4, 1), dtype=np.float32)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 5], dtype=np.int64)  # 类 2、5 各 1 条
    X2, y2 = drop_rare_classes(X, y, min_count=4)
    assert X2.shape[0] == 8
    assert set(y2.tolist()) == {0, 1}          # 重新连续编号
    assert y2.dtype == np.int64


def test_drop_rare_classes_keeps_all_when_threshold_met():
    X = np.zeros((6, 4, 1), dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    X2, y2 = drop_rare_classes(X, y, min_count=3)
    assert X2.shape[0] == 6 and set(y2.tolist()) == {0, 1}


# ---- describe：第1步统计表的唯一口径 ----

def test_describe_reports_real_shape_and_balance():
    X = np.zeros((10, 128, 3), dtype=np.float32)
    y = np.array([0] * 8 + [1] * 2, dtype=np.int64)
    d = describe(X, y)
    assert d["n_samples"] == 10
    assert d["seq_len"] == 128
    assert d["n_channels"] == 3
    assert d["n_classes"] == 2
    assert d["class_counts"] == [8, 2]
    assert d["imbalance_ratio"] == pytest.approx(4.0)
    assert d["chance"] == pytest.approx(0.5)   # 1/K，与 baseline_probe 的猜测线同义


# ---- canonical npz 往返 ----

def test_save_load_candidate_roundtrip(tmp_path):
    X = np.random.default_rng(2).normal(size=(7, 128, 2)).astype(np.float32)
    y = np.array([0, 1, 2, 0, 1, 2, 0], dtype=np.int64)
    save_candidate("Toy", X, y, {"fs": 50, "source_url": "http://x"}, root=tmp_path)
    X2, y2 = load_candidate("Toy", root=tmp_path)
    np.testing.assert_array_equal(X, X2)
    np.testing.assert_array_equal(y, y2)
    assert X2.dtype == np.float32 and y2.dtype == np.int64


def test_save_candidate_writes_meta_with_real_stats(tmp_path):
    import json

    X = np.zeros((5, 128, 4), dtype=np.float32)
    y = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    path = save_candidate("Toy", X, y, {"fs": 20}, root=tmp_path)
    meta = json.loads((tmp_path / "Toy.json").read_text(encoding="utf-8"))
    assert path.name == "Toy.npz"
    assert meta["fs"] == 20
    assert meta["n_samples"] == 5 and meta["n_channels"] == 4 and meta["n_classes"] == 2


def test_save_candidate_rejects_noncontiguous_labels(tmp_path):
    X = np.zeros((3, 128, 1), dtype=np.float32)
    y = np.array([0, 2, 2], dtype=np.int64)     # 缺 1 → 下游 SVM/切片假定 0..k-1
    with pytest.raises(ValueError):
        save_candidate("Toy", X, y, {}, root=tmp_path)


def test_load_candidate_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_candidate("NotThere", root=tmp_path)


# ---- 下载/解压：8 个 adapter 共用一份，避免各写各的（网络路径不进单测）----

def _make_zip(path, payload=b"hello"):
    import zipfile

    with zipfile.ZipFile(path, "w") as z:
        z.writestr("inner/a.txt", payload)


def test_fetch_archive_skips_download_when_cached(tmp_path):
    dest = tmp_path / "a.zip"
    _make_zip(dest)
    # URL 不可达：若函数尝试下载就会抛异常 → 命中缓存才可能通过
    got = fetch_archive("http://127.0.0.1:9/never", dest)
    assert got == dest and dest.exists()


def test_extract_archive_zip(tmp_path):
    src = tmp_path / "a.zip"
    _make_zip(src, b"payload")
    out = extract_archive(src, tmp_path / "ex")
    assert (out / "inner" / "a.txt").read_bytes() == b"payload"


def test_extract_archive_targz(tmp_path):
    import io
    import tarfile

    src = tmp_path / "a.tar.gz"
    with tarfile.open(src, "w:gz") as t:
        data = b"tarred"
        info = tarfile.TarInfo("inner/b.txt")
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    out = extract_archive(src, tmp_path / "ex")
    assert (out / "inner" / "b.txt").read_bytes() == b"tarred"


# ---- 抗混叠降采样（长序列候选的退路：保住整段时长而不是裁掉大半）----

def test_decimate_to_length_hits_target_length():
    X = np.random.default_rng(0).normal(size=(8, 3000, 1))
    Z = decimate_to_length(X, 256)
    assert Z.shape == (8, 256, 1)


def test_decimate_to_length_passthrough_when_already_short():
    X = np.random.default_rng(0).normal(size=(4, 128, 2))
    Z = decimate_to_length(X, 256)
    assert Z.shape == (4, 128, 2)          # 不上采样，短的原样返回


def test_decimate_to_length_preserves_low_frequency_content():
    # 1 Hz 正弦 @100 Hz、30 s → 降到 256 步（≈8.5 Hz）后仍应是同一条 1 Hz 正弦
    fs, T = 100, 3000
    t = np.arange(T) / fs
    X = np.sin(2 * np.pi * 1.0 * t)[None, :, None]
    Z = decimate_to_length(X, 256)
    # 降采样后每个周期约 8.5 个点，过零点数应仍对应 30 个周期（±1）
    zc = np.sum(np.diff(np.signbit(Z[0, :, 0])) != 0)
    assert 58 <= zc <= 62                   # 30 个周期 ≈ 60 次过零


def test_decimate_to_length_suppresses_aliasing_of_high_frequency():
    """抗混叠是这个函数存在的理由：裸切片会把高频折叠进低频。

    13 Hz（睡眠纺锤波频段）@100 Hz 降到 ≈8.5 Hz：Nyquist 4.27 Hz，13 Hz 必须被
    **滤掉**而不是折叠成 ~4.5 Hz 的假信号——后者会污染 delta 带，正是睡眠分期
    依赖的特征。故要求降采样后的能量显著低于裸切片。
    """
    fs, T = 100, 3000
    t = np.arange(T) / fs
    X = np.sin(2 * np.pi * 13.0 * t)[None, :, None]
    naive = X[:, :: T // 256, :][:, :256, :]          # 裸切片：混叠，能量几乎不减
    Z = decimate_to_length(X, 256)
    assert Z.std() < 0.25 * naive.std()


def test_decimate_to_length_keeps_dtype_and_finiteness():
    X = np.random.default_rng(1).normal(size=(3, 1000, 4)).astype(np.float32)
    Z = decimate_to_length(X, 256)
    assert Z.dtype == np.float32 and np.isfinite(Z).all()


def test_extract_archive_is_idempotent(tmp_path):
    src = tmp_path / "a.zip"
    _make_zip(src)
    out1 = extract_archive(src, tmp_path / "ex")
    out2 = extract_archive(src, tmp_path / "ex")
    assert out1 == out2 and (out2 / "inner" / "a.txt").exists()
