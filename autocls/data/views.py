"""输入视图（对比学习在哪个域上做）：时域 raw / 差分 diff / 频域 psd / 时频 spec。

动机：AutoCL 空间只在时域上做增强与对比；时频一致性（TF-C）、频域对比（CoST / BTSF）说明“在哪个域上对比”
本身是对比学习策略的一维，而且它的好坏强烈依赖数据集（平稳 / 节律信号如 EEG、振动、声学偏频域，形状类偏时域）。

所有视图都是逐样本的确定性变换（训练集 / 测试集可以分开变换，不引入跨样本泄漏），输入 (N, T, C) float32，
输出 (N, T', C') float32；变换后由调用方用 TRAIN 统计量逐通道再标准化（standardize_train）。
"""
from __future__ import annotations

import numpy as np

VIEWS = ("raw", "diff", "psd", "spec")
PSD_MAX_BINS = 256          # 频域序列长度上限（超过则相邻频点在功率域取均值池化）
SPEC_MAX_WIN = 256          # STFT 窗长上限
SPEC_MIN_WIN = 8
SPEC_MAX_BANDS = 16         # 每通道频带数上限
SPEC_MAX_CHANNELS = 4096    # C × 频带 的上限（超过则减少频带，最少 1 个）
_EPS = 1e-8


def _pool_bins(P: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    """沿 axis 把 F 个频点均分成 n_out 组取均值（功率域）。"""
    F = P.shape[axis]
    if F <= n_out:
        return P
    edges = np.linspace(0, F, n_out + 1).round().astype(int)
    parts = [np.take(P, np.arange(edges[i], max(edges[i + 1], edges[i] + 1)), axis=axis).mean(axis=axis, keepdims=True)
             for i in range(n_out)]
    return np.concatenate(parts, axis=axis)


def _diff(X: np.ndarray) -> np.ndarray:
    D = np.zeros_like(X)
    D[:, 1:] = X[:, 1:] - X[:, :-1]
    return np.concatenate([X, D], axis=-1)


def _psd(X: np.ndarray) -> np.ndarray:
    """逐样本逐通道：去均值 → Hann 窗 → |rfft|² → 去直流 → 池化到 ≤ PSD_MAX_BINS → log10。频率轴当序列轴。"""
    X = np.asarray(X, dtype=np.float64)
    T = X.shape[1]
    w = np.hanning(T)[None, :, None]
    Z = np.fft.rfft((X - X.mean(axis=1, keepdims=True)) * w, axis=1)
    P = (np.abs(Z) ** 2)[:, 1:]                     # (N, T//2 (+1), C)，去直流
    P = P[:, : T // 2]
    P = _pool_bins(P, PSD_MAX_BINS, axis=1)
    return np.log10(P + _EPS).astype(np.float32)


def spec_window(T: int) -> int:
    n = int(2 ** round(np.log2(max(SPEC_MIN_WIN, T / 8.0))))
    return int(min(max(n, SPEC_MIN_WIN), SPEC_MAX_WIN, T))


def _spec(X: np.ndarray) -> np.ndarray:
    """STFT 对数功率：窗长 ≈ T/8（2 的幂，8..256），半重叠 Hann；频点去直流后池化成 B 个频带；
    输出 (N, 帧数, C·B)，通道顺序 = 通道主序（c0b0, c0b1, …）。"""
    X = np.asarray(X, dtype=np.float64)
    N, T, C = X.shape
    n = spec_window(T)
    hop = max(n // 2, 1)
    n_frames = 1 + (T - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(n_frames)[:, None]       # (frames, n)
    w = np.hanning(n)[None, None, :, None]
    seg = X[:, idx, :]                                                      # (N, frames, n, C)
    seg = (seg - seg.mean(axis=2, keepdims=True)) * w
    P = (np.abs(np.fft.rfft(seg, axis=2)) ** 2)[:, :, 1:]                   # (N, frames, n//2, C)
    B = int(min(SPEC_MAX_BANDS, P.shape[2], max(1, SPEC_MAX_CHANNELS // C)))
    P = _pool_bins(P, B, axis=2) if P.shape[2] > B else P
    B = P.shape[2]
    P = np.log10(P + _EPS)
    return np.transpose(P, (0, 1, 3, 2)).reshape(N, n_frames, C * B).astype(np.float32)


def apply_view(X: np.ndarray, view: str) -> np.ndarray:
    if view not in VIEWS:
        raise ValueError(f"未知 view={view!r}（可选 {VIEWS}）")
    if view == "raw":
        return X
    if view == "diff":
        return _diff(np.asarray(X, dtype=np.float32))
    if view == "psd":
        return _psd(X)
    return _spec(X)


def standardize_train(Xtr: np.ndarray, *others: np.ndarray):
    """TRAIN 上逐通道拟合均值 / 标准差，套用到所有数组（= 官方 scaler: train）。"""
    flat = np.asarray(Xtr, dtype=np.float64).reshape(-1, Xtr.shape[-1])
    mu, sd = flat.mean(0), flat.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    out = [((np.asarray(a, dtype=np.float64) - mu) / sd).astype(np.float32) for a in (Xtr,) + others]
    return tuple(out)


# ---------------------------------------------------------------- 视图地标（E 的一部分：每个视图在 TRAIN 上保住多少类别信息）
VIEW_PROFILE_COLS = tuple(f"vinfo_{v}" for v in VIEWS) + tuple(f"vgain_{v}" for v in VIEWS if v != "raw")


def view_profile(X: np.ndarray, y: np.ndarray, *, max_n: int = 2000, max_dim: int = 1024, seed: int = 0) -> dict:
    """每个视图：变换 → TRAIN 统计量逐通道标准化 → 展平（> max_dim 维时高斯随机投影）→ LOO-1NN 精度按 chance 归一（info_loo）。
    vgain_v = vinfo_v − vinfo_raw。只用传入的（TRAIN）样本与标签；样本 > max_n 时按类分层子采样。"""
    from .mech_features import info_loo
    X = np.asarray(X, np.float32); y = np.asarray(y)
    rng = np.random.default_rng(seed)
    if len(y) > max_n:
        keep = []
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            keep += rng.choice(idx, size=max(1, int(round(max_n * len(idx) / len(y)))), replace=False).tolist()
        keep = np.sort(np.array(keep)); X, y = X[keep], y[keep]
    out = {}
    for v in VIEWS:
        F = standardize_train(np.concatenate([apply_view(X[i:i + 256], v) for i in range(0, len(X), 256)], axis=0))[0]
        F = F.reshape(len(F), -1).astype(np.float32)
        if F.shape[1] > max_dim:
            P = np.random.default_rng(seed + 1).standard_normal((F.shape[1], max_dim)).astype(np.float32) / np.sqrt(max_dim)
            F = F @ P
        out[f"vinfo_{v}"] = float(info_loo(F, y))
    for v in VIEWS:
        if v != "raw":
            out[f"vgain_{v}"] = out[f"vinfo_{v}"] - out["vinfo_raw"]
    return out
