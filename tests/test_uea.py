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

from autocls.data.uea import (
    _parse_ts_tolerant,
    encode_labels,
    load_uea_cached,
    load_uea_merged,
    pad_and_zscore,
)

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


# ---- load_uea_merged：TRAIN+TEST 合并（路线 A 源池"整数据集当候选"口径）----

def test_load_uea_merged_concatenates_both_splits():
    name = "ERing"
    if not (CACHE / name / f"{name}_TRAIN.ts").exists():
        pytest.skip(f"{name} 缓存缺失")
    Xtr, ytr = load_uea_cached(name, split="TRAIN", cache_dir=CACHE)
    Xte, yte = load_uea_cached(name, split="TEST", cache_dir=CACHE)
    X, y = load_uea_merged(name, cache_dir=CACHE)
    assert X.shape[0] == len(ytr) + len(yte)           # 合并样本数 = TRAIN + TEST
    assert X.ndim == 3 and X.shape[2] == Xtr.shape[2]  # 通道数一致
    assert X.dtype == np.float32 and y.dtype == np.int64
    assert set(y.tolist()) == set(range(len(set(ytr.tolist()) | set(yte.tolist()))))
    assert not np.isnan(X).any()


def test_load_uea_merged_common_length_for_ragged_splits():
    # 不等长数据集两切分各自 L_max 可能不同（如 SpokenArabicDigits）——合并后必须
    # 对齐到统一 T（否则无法 concat / 后续切片崩）。
    name = "SpokenArabicDigits"
    if not (CACHE / name / f"{name}_TRAIN.ts").exists():
        pytest.skip(f"{name} 缓存缺失")
    X, y = load_uea_merged(name, cache_dir=CACHE)
    assert X.shape[0] == X.shape[0]                    # 单一等长块
    assert X.ndim == 3
    assert not np.isnan(X).any()


def test_load_uea_merged_crop_len():
    name = "Cricket"
    if not (CACHE / name / f"{name}_TRAIN.ts").exists():
        pytest.skip("Cricket 缓存缺失")
    X, _ = load_uea_merged(name, cache_dir=CACHE, crop_len=256)
    assert X.shape[1] == 256


# ---- 变长数据集经 cl_trainer 不崩 + 中性零填充（路线 A 验收 5：无显式 mask，
#      填充=z-score 后通道均值 0，探针实测 SpokenArabicDigits SNR 2.24 已证够用）----

# ---- 容错 .ts 解析（CharacterTrajectories 21/1422 例"同 case 内通道长度不一"，
#      aeon 拒载；回退：各通道截断到 case 内最短通道长，截断非填充不造假值）----

def test_parse_ts_tolerant_truncates_ragged_channels(tmp_path):
    ts = tmp_path / "R" / "R_TRAIN.ts"
    ts.parent.mkdir()
    ts.write_text(
        "@problemName R\n@classLabel true 1 2\n@data\n"
        "1.0,2.0,3.0:4.0,5.0:7\n"          # ragged：通道长 3 与 2 → 截断到 2
        "1.0,2.0:3.0,4.0:9\n",              # 齐整：通道长 2、2
        encoding="utf-8",
    )
    seqs, y = _parse_ts_tolerant(ts)
    assert len(seqs) == 2
    assert seqs[0].shape == (2, 2)          # 截断到 case 最短通道长 2
    assert seqs[1].shape == (2, 2)
    # 截断保留前缀真实值（通道 0 前 2 步 = 1,2；通道 1 = 4,5）
    np.testing.assert_allclose(seqs[0], [[1.0, 4.0], [2.0, 5.0]])
    assert list(y) == ["7", "9"]


def test_load_uea_merged_character_trajectories_via_fallback():
    # aeon 拒载 → uea 自动回退容错解析，CharacterTrajectories 可加载入源池
    name = "CharacterTrajectories"
    if not (CACHE / name / f"{name}_TRAIN.ts").exists():
        pytest.skip(f"{name} 缓存缺失")
    X, y = load_uea_merged(name, cache_dir=CACHE)
    assert X.shape[1] == 182 and X.shape[2] == 3   # pad 到 L_max=182、3 通道
    assert set(y.tolist()) == set(range(20))       # 20 类
    assert not np.isnan(X).any()


@pytest.mark.parametrize("name,pad_L", [
    ("SpokenArabicDigits", 93),        # 不等长 4-93 → pad 93
    ("CharacterTrajectories", 182),    # 不等长 60-182 → pad 182（经容错回退加载）
])
def test_variable_length_pretrain_no_crash(name, pad_L, make_cls):
    if not (CACHE / name / f"{name}_TRAIN.ts").exists():
        pytest.skip(f"{name} 缓存缺失")
    from autocls.contrastive.cl_trainer import pretrain_encoder
    from autocls.encoders.registry import build_encoder

    X, y = load_uea_merged(name, cache_dir=CACHE)
    assert X.shape[1] == pad_L
    assert not np.isnan(X).any()                       # 填充不产 NaN
    # 中性零填充存在：变长数据集有整段全零时间步（z-score 后 = 通道均值 0）
    padded_steps = (X == 0).all(axis=2).sum()
    assert padded_steps > 0, "变长数据集应有零填充步"
    # 取 2 类一批喂 cl_trainer：变长 + 填充步不致崩、损失有限（填充步不主导 → 不发散）
    cls0 = np.unique(y)[:2]
    idx = np.where(np.isin(y, cls0))[0][:96]
    enc = build_encoder(input_dims=X.shape[-1])
    losses = pretrain_encoder(
        enc, make_cls(), X[idx].astype(np.float32),
        max_iters=20, batch_size=32, device="cpu", seed=0,
    )
    assert losses and all(np.isfinite(v) for v in losses)


# ---- 全量协议对齐（M8）：load_uea_pair(scaler="split"|"train") ----
# "split"（既有口径）= 两次 load_uea_cached 逐位相同；"train"（TS2Vec 官方
# datautils.load_UEA 语义）= 统计量只在 TRAIN 真实步上拟合、TEST 用同一统计量、
# 两切分共用目标长度、标签在两切分并集上编码。pad_and_zscore 拆成
# _fit_channel_stats + _apply_pad_zscore 后组合输出必须逐位不变（金标 md5）。

import hashlib  # noqa: E402

from autocls.data import uea as uea_mod  # noqa: E402
from autocls.data.uea import load_uea_pair  # noqa: E402


def _golden_ragged_seqs():
    rng = np.random.default_rng(20260903)
    seqs = [rng.normal(loc=3.0, scale=2.0, size=(L, 3)) for L in (7, 12, 5, 12, 9)]
    seqs[2][:, 1] = 4.0
    return seqs


def test_pad_and_zscore_refactor_bitwise_golden():
    # 2026-09-03 拆分前捕获（scratchpad）：输出逐位不变是历史缓存的复现红线
    seqs = _golden_ragged_seqs()
    X = pad_and_zscore(seqs)
    Xc = pad_and_zscore(seqs, crop_len=8)
    assert X.shape == (5, 12, 3) and X.dtype == np.float32
    assert hashlib.md5(X.tobytes()).hexdigest() == "be7af9a43b420a4d49231048257ba205"
    assert Xc.shape == (5, 8, 3)
    assert hashlib.md5(Xc.tobytes()).hexdigest() == "9bca18e6ec69c9b758130a4dbb757d10"


def _fake_raw(monkeypatch, train, test):
    """monkeypatch _load_raw_seqs：不依赖 aeon/.ts 文件的合成两切分。"""
    def fake(name, split, cache_dir):
        seqs, y = train if split == "TRAIN" else test
        return [np.asarray(s, dtype=np.float64) for s in seqs], np.asarray(y)
    monkeypatch.setattr(uea_mod, "_load_raw_seqs", fake)


def _synthetic_splits(seed=0):
    rng = np.random.default_rng(seed)
    tr = ([rng.normal(loc=2.0, scale=3.0, size=(5, 2)) for _ in range(12)],
          ["a", "b", "c"] * 4)
    te = ([rng.normal(loc=5.0, scale=1.0, size=(7, 2)) for _ in range(6)],
          ["a", "b"] * 3)                       # TEST 缺类 "c"
    return tr, te


def test_load_uea_pair_split_equals_two_cached(monkeypatch, tmp_path):
    _fake_raw(monkeypatch, *_synthetic_splits())
    Xtr, ytr, Xte, yte = load_uea_pair("Fake", cache_dir=tmp_path, scaler="split")
    Xtr2, ytr2 = load_uea_cached("Fake", "TRAIN", cache_dir=tmp_path)
    Xte2, yte2 = load_uea_cached("Fake", "TEST", cache_dir=tmp_path)
    assert np.array_equal(Xtr, Xtr2) and np.array_equal(Xte, Xte2)
    assert Xtr.tobytes() == Xtr2.tobytes() and Xte.tobytes() == Xte2.tobytes()
    assert np.array_equal(ytr, ytr2) and np.array_equal(yte, yte2)


def test_load_uea_pair_split_matches_real_cache_bitwise():
    name = "Handwriting"
    if not (CACHE / name / f"{name}_TRAIN.ts").exists():
        pytest.skip("Handwriting 缓存缺失")
    Xtr, ytr, Xte, yte = load_uea_pair(name, cache_dir=CACHE, scaler="split")
    Xtr2, ytr2 = load_uea_cached(name, "TRAIN", cache_dir=CACHE)
    Xte2, yte2 = load_uea_cached(name, "TEST", cache_dir=CACHE)
    assert Xtr.tobytes() == Xtr2.tobytes() and Xte.tobytes() == Xte2.tobytes()
    assert np.array_equal(ytr, ytr2) and np.array_equal(yte, yte2)


def test_load_uea_pair_train_scaler_stats_from_train_only(monkeypatch, tmp_path):
    tr, te = _synthetic_splits()
    _fake_raw(monkeypatch, tr, te)
    Xtr, ytr, Xte, yte = load_uea_pair("Fake", cache_dir=tmp_path, scaler="train")
    # 目标长度 = 两切分真实长度最大值（7），TRAIN 也 pad 到 7
    assert Xtr.shape == (12, 7, 2) and Xte.shape == (6, 7, 2)
    # TRAIN 真实步 z-score 后均值≈0/std≈1
    real_tr = Xtr[:, :5].reshape(-1, 2)
    assert np.allclose(real_tr.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(real_tr.std(axis=0), 1.0, atol=1e-4)
    # TEST 用 TRAIN 统计量：TEST 本身 loc=5/scale=1 ≠ TRAIN 的 2/3 → 不会是 0 均值
    real_te = Xte[:, :7].reshape(-1, 2)
    stacked = np.concatenate([np.asarray(s) for s in tr[0]], axis=0)
    mean, std = stacked.mean(axis=0), stacked.std(axis=0)
    expect = np.concatenate([(np.asarray(s) - mean) / std for s in te[0]], axis=0)
    assert np.allclose(real_te, expect.astype(np.float32), atol=1e-6)
    assert not np.allclose(real_te.mean(axis=0), 0.0, atol=1e-2)


def test_load_uea_pair_train_labels_consistent_across_splits(monkeypatch, tmp_path):
    tr, te = _synthetic_splits()
    _fake_raw(monkeypatch, tr, te)
    _Xtr, ytr, _Xte, yte = load_uea_pair("Fake", cache_dir=tmp_path, scaler="train")
    # 并集编码：a→0, b→1, c→2；TEST 只有 a/b 但 id 与 TRAIN 一致（不重排成 0/1）
    assert ytr.tolist() == [0, 1, 2] * 4
    assert yte.tolist() == [0, 1] * 3
    assert ytr.dtype == np.int64 and yte.dtype == np.int64


def test_load_uea_pair_train_crop_len(monkeypatch, tmp_path):
    _fake_raw(monkeypatch, *_synthetic_splits())
    Xtr, _ytr, Xte, _yte = load_uea_pair("Fake", cache_dir=tmp_path, scaler="train", crop_len=4)
    assert Xtr.shape[1] == 4 and Xte.shape[1] == 4


def test_load_uea_pair_unknown_scaler_raises(monkeypatch, tmp_path):
    _fake_raw(monkeypatch, *_synthetic_splits())
    with pytest.raises(ValueError):
        load_uea_pair("Fake", cache_dir=tmp_path, scaler="global")
