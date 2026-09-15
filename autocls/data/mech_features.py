"""数据集级机制特征：只用目标的训练划分（允许标签），零预训练，与 CLS 字段同坐标。

自 wp11-mech 分支移植（`zscore_tc` / `_nn_stats` / `hardness_1nn` / `label_retention` / `chan_corr` / `compute_features` 逐字），
并加入 WP14-HSPACE 阶段 2 已过留出的"信息量差"口径与 WP16 阶段 1 A 臂的**响应画像**（`response_profile`）：

- 难度 ``hardness_1nn``：逐 (样本, 通道) 沿时间 z-score 后展平的留一 1-NN 精度按 chance 归一，H = 1 − (acc − chance)/(1 − chance)。
- 标签保持率（两套口径）：
  * ``label_retention``（wp11）：查询 = 增强后样本、参考 = 未增强训练集（排除自身），平均类别边际之比 / 精度之比；
  * ``retention_info``（WP14 v2，`ret_<op> = info_after − info_raw`，``info = clip((LOO-1NN acc − chance)/(1 − chance), −1, 1)``，
    特征逐维标准化）。``form="within"``：变换后集合内 LOO（确定性输入变换——部署时训练 / 评估同一前向）；
    ``form="cross"``：查询 = op(x)、参考 = 未变换 X 排除自身（随机增强 / 视图变换——只有一个视图被变换，锚不变）。
- 无标签失真 ``distortion_cos``：x 与 op(x) 展平后的平均余弦。
- 结构 ``linear_fit`` / ``trend_strength`` / ``trend_info`` / ``level_scale_info``（wp14 51 号逐字）。
- 确定性变换 ``deterministic_op(mode)``：inst_norm / detrend / diff1 / lowpass25 / highpass05（wp14 ``apply_input_transform`` 同式）
  与视图变换 polarity（−x）/ time_reverse，统一成增强 op 契约 ``op(x, p, generator) -> Tensor``。

铁律 7：本模块是任务侧模块，``compute_features`` / ``response_profile`` 按 ``task_type`` 显式分发（现只实现分类）；
任务无关模块（space / contrastive / encoders / comparator / search）不得 import 本模块。
铁律 6 / 泄漏：调用方只能传训练划分（UCR：``split_task_train_eval`` 的 train 侧；UEA：官方 TRAIN）。
torch 与增强 op 惰性导入，保持 ``autocls.data`` 包 import 时无 torch 依赖。
"""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np

from ..task_types import TASK_CLASSIFICATION, validate_task_type

OpLike = Union[str, Callable[..., Any]]

# WP14 dilated_cnn 的常数（逐字）：inst_norm eps、频带截止
INST_NORM_EPS = 1e-5
LOWPASS_KEEP_FRAC = 0.25
HIGHPASS_CUT_FRAC = 0.05
_BAND_EPS = 1e-9
DETERMINISTIC_MODES = ("inst_norm", "detrend", "diff1", "lowpass25", "highpass05", "polarity", "time_reverse")

# 响应画像的 op 表（协议 §2）：p = 档位（确定性 op 记 0），form = 保持率形式，sim = 是否进无标签失真块（polarity 恒 −1 不进）
RESPONSE_OPS: dict[str, dict[str, Any]] = {
    "jitter": {"p": 0.5, "form": "cross", "sim": True},
    "rescale": {"p": 0.5, "form": "cross", "sim": True},
    "point_mask": {"p": 0.5, "form": "cross", "sim": True},
    "freq_mask": {"p": 0.5, "form": "cross", "sim": True},
    "polarity": {"p": 0.0, "form": "cross", "sim": False},
    "time_reverse": {"p": 0.0, "form": "cross", "sim": True},
    "inst_norm": {"p": 0.0, "form": "within", "sim": True},
    "detrend": {"p": 0.0, "form": "within", "sim": True},
    "highpass05": {"p": 0.0, "form": "within", "sim": True},
    "lowpass25": {"p": 0.0, "form": "within", "sim": True},
    "diff1": {"p": 0.0, "form": "within", "sim": True},
}
STRUCTURAL_COLS = ("hardness_1nn", "level_scale_info", "trend_info", "trend_strength")
RESPONSE_PROFILE_COLS: tuple[str, ...] = (
    tuple(f"ret_{op}" for op in RESPONSE_OPS)
    + tuple(f"sim_{op}" for op, spec in RESPONSE_OPS.items() if spec["sim"])
    + STRUCTURAL_COLS
)
RESPONSE_SEEDS = (0, 1, 2)


# ---------------------------------------------------------------- 原始空间几何
def zscore_tc(X: np.ndarray) -> np.ndarray:
    """(N, T, C) 逐 (样本, 通道) 沿时间 z-score（= mx4_mask/mech_probe._zs，eps 1e-8）。"""
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mu) / sd


def _nn_stats(Fq: np.ndarray, Fr: np.ndarray, y: np.ndarray, *, exclude_self: bool, chunk: int) -> tuple[int, float]:
    """(1-NN 同类命中数, 平均类别边际)。边际_i = (d_异类最近 − d_同类最近)/(d_异类最近 + d_同类最近) ∈ [−1, 1]（欧氏距离，
    不含自身）——1-NN 精度的连续版（边际 > 0 ⇔ 1-NN 命中），无并列。"""
    Fq = np.asarray(Fq, dtype=np.float32)
    Fr = np.asarray(Fr, dtype=np.float32)
    y = np.asarray(y)
    n = Fq.shape[0]
    sq_q = (Fq ** 2).sum(axis=1)
    sq_r = (Fr ** 2).sum(axis=1)
    hits = 0
    margins = np.zeros(n, dtype=np.float64)
    for s in range(0, n, int(chunk)):
        e = min(n, s + int(chunk))
        d = sq_q[s:e, None] + sq_r[None, :] - 2.0 * Fq[s:e] @ Fr.T
        if exclude_self:
            d[np.arange(e - s), np.arange(s, e)] = np.inf
        nn = d.argmin(axis=1)
        hits += int((y[nn] == y[s:e]).sum())
        same = (y[None, :] == y[s:e, None])
        d_same = np.sqrt(np.maximum(np.where(same, d, np.inf).min(axis=1), 0.0))
        d_other = np.sqrt(np.maximum(np.where(~same, d, np.inf).min(axis=1), 0.0))
        margins[s:e] = (d_other - d_same) / (d_other + d_same + 1e-12)
    return hits, float(np.mean(margins))


def _nn_hits(Fq: np.ndarray, Fr: np.ndarray, y: np.ndarray, *, exclude_self: bool, chunk: int) -> int:
    return _nn_stats(Fq, Fr, y, exclude_self=exclude_self, chunk=chunk)[0]


def nn1_loo_acc(F: np.ndarray, y: np.ndarray, *, chunk: int = 512) -> float:
    """留一 1-NN 精度（自身距离置 inf；分块矩阵乘，chunk 只影响内存不影响结果）。"""
    n = np.asarray(F).shape[0]
    return _nn_hits(F, F, y, exclude_self=True, chunk=chunk) / n


def nn1_cross_acc(Fq: np.ndarray, Fr: np.ndarray, y: np.ndarray, *, exclude_self: bool = True,
                  chunk: int = 512) -> float:
    """查询集 Fq（第 i 行是样本 i 的变换）在参考集 Fr（未变换）里的 1-NN 标签一致率；exclude_self 把 (i, i) 置 inf。"""
    n = np.asarray(Fq).shape[0]
    return _nn_hits(Fq, Fr, y, exclude_self=exclude_self, chunk=chunk) / n


def nn_margin(Fq: np.ndarray, Fr: np.ndarray, y: np.ndarray, *, exclude_self: bool = True, chunk: int = 512) -> float:
    """平均类别边际（连续版 1-NN）：查询 Fq 对参考 Fr 的 (d_other − d_same)/(d_other + d_same) 均值。"""
    return _nn_stats(Fq, Fr, y, exclude_self=exclude_self, chunk=chunk)[1]


def majority_chance(y: np.ndarray) -> float:
    """训练划分多数类占比（作 chance；uea30.yaml 的 chance 是 TEST 侧，不用）。"""
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / len(y))


def info_norm(acc: float, chance: float) -> float:
    """1-NN 精度按 chance 归一，clip [−1, 1]（wp14 51 号 `_info` 的归一半边）。"""
    return float(np.clip((float(acc) - float(chance)) / max(1.0 - float(chance), 1e-9), -1.0, 1.0))


def _standardize_cols(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    sd = F.std(axis=0)
    return (F - F.mean(axis=0)) / np.where(sd > 1e-12, sd, 1.0)


def info_loo(F: np.ndarray, y: np.ndarray, *, chunk: int = 512) -> float:
    """逐维标准化后 LOO-1NN 精度按 chance 归一（wp14 51 号 `_info` 逐字）。"""
    y = np.asarray(y)
    return info_norm(nn1_loo_acc(_standardize_cols(F), y, chunk=chunk), majority_chance(y))


# ---------------------------------------------------------------- 变换 op（增强 op 契约：op(x, p, generator) -> Tensor）
def _rfft_bins_kept(n_freq: int, mode: str):
    import torch
    n_freq = int(n_freq)
    k = torch.arange(n_freq)
    if mode == "lowpass25":
        return k <= math.floor(LOWPASS_KEEP_FRAC * (n_freq - 1) + _BAND_EPS)
    if mode == "highpass05":
        k_cut = max(1, math.ceil(HIGHPASS_CUT_FRAC * (n_freq - 1) - _BAND_EPS))   # DC 必去
        return k >= k_cut
    raise ValueError(f"未知频带模式 {mode!r}")


def apply_deterministic(x, mode: str):
    """(B, T, C) 张量的确定性变换（wp14 `apply_input_transform` 同式）+ 视图变换 polarity / time_reverse。"""
    import torch
    if mode == "inst_norm":
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + INST_NORM_EPS)
    if mode == "detrend":
        T = x.shape[1]
        t = torch.arange(T, dtype=x.dtype, device=x.device)
        t = t - t.mean()
        denom = (t * t).sum().clamp_min(1e-12)
        mean = x.mean(dim=1, keepdim=True)
        slope = (x * t.view(1, T, 1)).sum(dim=1, keepdim=True) / denom
        return x - mean - slope * t.view(1, T, 1)
    if mode == "diff1":
        d = x[:, 1:] - x[:, :-1]
        return torch.cat([torch.zeros_like(x[:, :1]), d], dim=1)
    if mode in ("lowpass25", "highpass05"):
        T = x.shape[1]
        xf = torch.fft.rfft(x, dim=1)
        keep = _rfft_bins_kept(xf.shape[1], mode).to(device=xf.device).view(1, -1, 1)
        return torch.fft.irfft(xf * keep, n=T, dim=1).to(x.dtype)
    if mode == "polarity":
        return -x
    if mode == "time_reverse":
        return torch.flip(x, dims=[1])
    raise ValueError(f"未知确定性变换 {mode!r}（可选 {DETERMINISTIC_MODES}）")


def deterministic_op(mode: str) -> Callable[..., Any]:
    """把确定性变换包成增强 op 契约（p 与 generator 忽略）。"""
    if mode not in DETERMINISTIC_MODES:
        raise ValueError(f"未知确定性变换 {mode!r}（可选 {DETERMINISTIC_MODES}）")
    return lambda x, p, generator=None, _m=mode: apply_deterministic(x, _m)


def _resolve_op(op: OpLike) -> Callable[..., Any]:
    if callable(op):
        return op
    if op in DETERMINISTIC_MODES:
        return deterministic_op(op)
    from ..contrastive import augmentations as A   # 惰性：与训练用的是同一个 op
    fn = getattr(A, op, None)
    if fn is None or not callable(fn):
        raise KeyError(f"未知增强 op {op!r}（合法：augmentations 里的函数名或 {DETERMINISTIC_MODES}）")
    return fn


def _apply_op(op: OpLike, Z: np.ndarray, p: float, seed: int) -> np.ndarray:
    """跑一次 op（返回 Tensor 或 (Tensor, …) 元组都接受），保形检查，回 float32 数组。"""
    import torch
    fn = _resolve_op(op)
    gen = torch.Generator().manual_seed(int(seed))
    out = fn(torch.from_numpy(np.ascontiguousarray(np.asarray(Z, dtype=np.float32))), float(p), generator=gen)
    x_aug = out[0] if isinstance(out, tuple) else out
    Za = np.asarray(x_aug.detach().cpu().numpy(), dtype=np.float32)
    if Za.shape != np.shape(Z):
        raise ValueError(f"增强 op 须保形：{np.shape(Z)} → {Za.shape}")
    return Za


# ---------------------------------------------------------------- 特征
def hardness_1nn(X: np.ndarray, y: np.ndarray, *, chance: Optional[float] = None, chunk: int = 512) -> dict[str, float]:
    """H = 1 − (LOO-1NN acc − chance)/(1 − chance)，截到 [0, 1]。"""
    y = np.asarray(y)
    ch = float(majority_chance(y) if chance is None else chance)
    Z = zscore_tc(X)
    acc = nn1_loo_acc(Z.reshape(Z.shape[0], -1), y, chunk=chunk)
    denom = max(1.0 - ch, 1e-12)
    h = float(np.clip(1.0 - (acc - ch) / denom, 0.0, 1.0))
    return {"hardness": h, "nn1_base_acc": float(acc), "chance": ch}


def label_retention(X: np.ndarray, y: np.ndarray, op: OpLike, p: float, *, seed: int = 0,
                    chance: Optional[float] = None, min_margin: float = 1e-9, chunk: int = 512) -> dict[str, Any]:
    """标签保持率（wp11 两种口径）。

    - ``retention``（主，连续）：R_f = clip(margin_aug / margin_base, −1, 1)，margin = 平均类别边际
      (d_异类最近 − d_同类最近)/(d_异类最近 + d_同类最近)，查询 = 每个样本经 ``op(x, p, generator)`` 变换一次，
      参考 = 未增强训练集（排除自身）；margin_base = 未增强的留一边际。
    - ``retention_acc``（副，精度口径）：clip((acc_aug − chance)/(acc_base − chance), 0, 1)。
    op 是增强函数名 / 确定性变换名或同契约的可调用；p=0 → 随机 op 恒等 → 两口径都恒等于 1。
    margin_base ≤ 0 或 acc_base − chance < min_margin（类结构在原始空间不可见）→ 对应口径 nan。
    """
    y = np.asarray(y)
    ch = float(majority_chance(y) if chance is None else chance)
    Z = zscore_tc(X)
    N = Z.shape[0]
    Fr = Z.reshape(N, -1)
    hits_b, margin_base = _nn_stats(Fr, Fr, y, exclude_self=True, chunk=chunk)
    acc_base = hits_b / N
    Za = _apply_op(op, Z, p, seed)
    hits_a, margin_aug = _nn_stats(Za.reshape(N, -1), Fr, y, exclude_self=True, chunk=chunk)
    acc_aug = hits_a / N
    denom = acc_base - ch
    ret_acc = float("nan") if denom < float(min_margin) else float(np.clip((acc_aug - ch) / denom, 0.0, 1.0))
    ret = float("nan") if margin_base <= float(min_margin) else float(np.clip(margin_aug / margin_base, -1.0, 1.0))
    return {"retention": ret, "retention_acc": ret_acc, "margin_aug": float(margin_aug), "margin_base": float(margin_base),
            "acc_aug": float(acc_aug), "acc_base": float(acc_base), "chance": ch}


def retention_info(X: np.ndarray, y: np.ndarray, op: OpLike, p: float, *, form: str, seeds: Sequence[int] = RESPONSE_SEEDS,
                   chunk: int = 512) -> dict[str, float]:
    """WP14 v2 口径的保持率：``delta = info_after − info_raw``（X 原样，不逐样本 z-score；信息量里逐维标准化）。

    form="within"：info(op(X)) 在变换后集合内 LOO（确定性变换只跑 seeds[0]，随机 op 取多种子均值）；
    form="cross"：查询 = op(x)、参考 = 未变换 X（排除自身），逐维标准化用参考集的 μ/σ 同时作用于查询。
    """
    if form not in ("within", "cross"):
        raise ValueError(f"form 须为 within / cross，收到 {form!r}")
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    N = X.shape[0]
    ch = majority_chance(y)
    Fr = X.reshape(N, -1).astype(float)
    mu, sd = Fr.mean(axis=0), Fr.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Frs = (Fr - mu) / sd
    info_raw = info_norm(nn1_loo_acc(Frs, y, chunk=chunk), ch)
    vals = []
    for s in seeds:
        Fa = _apply_op(op, X, p, s).reshape(N, -1).astype(float)
        if form == "within":
            acc = nn1_loo_acc(_standardize_cols(Fa), y, chunk=chunk)
        else:
            acc = nn1_cross_acc((Fa - mu) / sd, Frs, y, exclude_self=True, chunk=chunk)
        vals.append(info_norm(acc, ch))
    info_after = float(np.mean(vals))
    return {"info_raw": float(info_raw), "info_after": info_after, "delta": float(np.clip(info_after - info_raw, -1.0, 1.0)),
            "form": form, "n_seeds": len(vals)}


def distortion_cos(X: np.ndarray, op: OpLike, p: float, *, seed: int = 0) -> float:
    """x 与 op(x) 展平后的平均余弦（无标签失真；零向量按 0 处理）。"""
    X = np.asarray(X, dtype=np.float32)
    N = X.shape[0]
    A = X.reshape(N, -1).astype(float)
    B = _apply_op(op, X, p, seed).reshape(N, -1).astype(float)
    na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
    denom = na * nb
    cos = np.where(denom > 1e-12, (A * B).sum(axis=1) / np.where(denom > 1e-12, denom, 1.0), 0.0)
    return float(np.clip(cos.mean(), -1.0, 1.0))


def linear_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐样本逐通道最小二乘直线：(slope, mean, R²)，各 (N, C)。R² = 趋势解释的方差份额（常数序列记 0）。"""
    X = np.asarray(X, dtype=float)
    N, T, C = X.shape
    t = np.arange(T, dtype=float)
    tc = t - t.mean()
    stt = float((tc * tc).sum())
    mean = X.mean(axis=1)
    xc = X - mean[:, None, :]
    slope = np.einsum("ntc,t->nc", xc, tc) / stt
    ss_tot = (xc * xc).sum(axis=1)
    ss_trend = slope * slope * stt
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_tot > 0, ss_trend / ss_tot, 0.0)
    return slope, mean, np.clip(r2, 0.0, 1.0)


def trend_strength(X: np.ndarray) -> float:
    return float(linear_fit(X)[2].mean())


def trend_info(X: np.ndarray, y: np.ndarray) -> float:
    slope, mean, _ = linear_fit(X)
    return info_loo(np.concatenate([slope, mean], axis=1), y)


def level_scale_info(X: np.ndarray, y: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    return info_loo(np.concatenate([X.mean(axis=1), X.std(axis=1)], axis=1), y)


def chan_corr(X: np.ndarray, *, n_sub: int = 200, seed: int = 0) -> float:
    """通道间平均 |Pearson|（redundancy_probe 公式逐字）：最多 n_sub 个样本、常数通道跳过、C=1 → nan。"""
    X = np.asarray(X)
    N, _T, C = X.shape
    if C < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=min(N, int(n_sub)), replace=False)
    Xs = np.nan_to_num(X[idx])
    vals = []
    for x in Xs:
        ok = x.std(axis=0) > 1e-6
        if ok.sum() < 2:
            continue
        cc = np.corrcoef(x[:, ok].T)
        iu = np.triu_indices(int(ok.sum()), 1)
        v = np.abs(cc[iu])
        v = v[np.isfinite(v)]
        if v.size:
            vals.append(float(v.mean()))
    return float(np.mean(vals)) if vals else float("nan")


def _nan_to_none(v: Optional[float]) -> Optional[float]:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def compute_features(X: np.ndarray, y: np.ndarray, *, task_type: str, ops: Mapping[str, float],
                     seed: int = 0, chunk: int = 512) -> dict[str, Any]:
    """一个数据集（训练划分）的 wp11 机制特征；按 task_type 显式分发（铁律 7），现只实现分类。

    ``ops`` = {增强名: 档位}，每项算一个标签保持率，键写作 ``"<op>=<p>"``；C=1 时 channel_drop 恒等 → None。
    """
    validate_task_type(task_type)
    if task_type != TASK_CLASSIFICATION:
        raise NotImplementedError(f"task_type={task_type!r} 的机制特征未定义（难度 / 保持率都以标签一致率定义）")
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    N, T, C = X.shape
    h = hardness_1nn(X, y, chunk=chunk)
    ret: dict[str, Optional[float]] = {}
    ret_acc: dict[str, Optional[float]] = {}
    margin_base: Optional[float] = None
    for op, p in ops.items():
        if op == "channel_drop" and C < 2:
            ret[f"{op}={p}"] = None
            ret_acc[f"{op}={p}"] = None
            continue
        r = label_retention(X, y, op, float(p), seed=seed, chance=h["chance"], chunk=chunk)
        ret[f"{op}={p}"] = _nan_to_none(r["retention"])
        ret_acc[f"{op}={p}"] = _nan_to_none(r["retention_acc"])
        margin_base = r["margin_base"]
    return {"n": int(N), "T": int(T), "C": int(C), "K": int(len(np.unique(y))), "chance": h["chance"],
            "log10_T": float(np.log10(T)), "hardness_1nn": h["hardness"], "nn1_base_acc": h["nn1_base_acc"],
            "margin_base": margin_base, "retention": ret, "retention_acc": ret_acc,
            "chan_corr": _nan_to_none(chan_corr(X, seed=seed)), "feature_seed": int(seed)}


def response_profile(X: np.ndarray, y: np.ndarray, *, task_type: str, seeds: Sequence[int] = RESPONSE_SEEDS,
                     chunk: int = 512) -> dict[str, Any]:
    """WP16 阶段 1 A 臂的响应画像 E_A（协议 §2）：``RESPONSE_PROFILE_COLS`` 25 维 + 诊断字段。

    按 task_type 显式分发（铁律 7），现只实现分类。X 原样（不逐样本 z-score；hardness_1nn 内部自己 z-score）。
    """
    validate_task_type(task_type)
    if task_type != TASK_CLASSIFICATION:
        raise NotImplementedError(f"task_type={task_type!r} 的响应画像未定义（保持率以标签一致率定义）")
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    N, T, C = X.shape
    out: dict[str, Any] = {"n": int(N), "T": int(T), "C": int(C), "K": int(len(np.unique(y))), "chance": majority_chance(y),
                           "seeds": [int(s) for s in seeds]}
    info_raw = None
    for op, spec in RESPONSE_OPS.items():
        use_seeds = tuple(seeds) if spec["p"] > 0 else (int(seeds[0]),)
        r = retention_info(X, y, op, spec["p"], form=spec["form"], seeds=use_seeds, chunk=chunk)
        out[f"ret_{op}"] = r["delta"]
        out[f"info_{op}"] = r["info_after"]
        info_raw = r["info_raw"]
        if spec["sim"]:
            out[f"sim_{op}"] = distortion_cos(X, op, spec["p"], seed=int(seeds[0]))
    out["info_raw"] = info_raw
    h = hardness_1nn(X, y, chunk=chunk)
    out["hardness_1nn"] = h["hardness"]
    out["nn1_base_acc"] = h["nn1_base_acc"]
    out["level_scale_info"] = level_scale_info(X, y)
    out["trend_info"] = trend_info(X, y)
    out["trend_strength"] = trend_strength(X)
    return out


# ---------------------------------------------------------------- 转导型响应画像（WP16 T 检验，results/tclsc_s1et/PROTOCOL.md §2）
REP_PROFILE_COLS: tuple[str, ...] = (
    tuple(f"rep_{op}" for op in RESPONSE_OPS)
    + tuple(f"rcos_{op}" for op, spec in RESPONSE_OPS.items() if spec["sim"])
    + ("rinfo",)
)


def rep_cols_at(budget: int) -> tuple[str, ...]:
    """某截点的列名：`rep_<op>@b` / `rcos_<op>@b` / `rinfo@b`。"""
    return tuple(f"{c}@{int(budget)}" for c in REP_PROFILE_COLS)


def torch_encode_fn(encoder, device: Optional[str] = None, batch_size: int = 128) -> Callable[[np.ndarray], np.ndarray]:
    """nn.Module (B, T, C)→(B, T, d) → 编码函数 (N, T, C)→(N, d)（时间维 max-pool，与 R' 评估同：downstream.linear_eval.encode_features）。"""
    from ..downstream.linear_eval import encode_features
    return lambda X: encode_features(encoder, np.asarray(X, np.float32), device=device, batch_size=batch_size)


def info_cross_rep(R0: np.ndarray, Rp: np.ndarray, y: np.ndarray, *, chunk: int = 512) -> float:
    """扰动样本 Rp[i] 在未扰动库 R0（排除 i）里的 1-NN 类别命中，逐维标准化用 R0 的 μ/σ，chance 归一。"""
    R0 = np.asarray(R0, float); Rp = np.asarray(Rp, float)
    mu, sd = R0.mean(0), R0.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    y = np.asarray(y)
    return info_norm(nn1_cross_acc((Rp - mu) / sd, (R0 - mu) / sd, y, exclude_self=True, chunk=chunk), majority_chance(y))


def rep_profile(encode_fn: Callable[[np.ndarray], np.ndarray], X: np.ndarray, y: np.ndarray, *, task_type: str,
                seeds: Sequence[int] = RESPONSE_SEEDS, chunk: int = 512) -> dict[str, Any]:
    """转导型响应画像：同一套 RESPONSE_OPS 在**表示空间**里的响应（22 维 REP_PROFILE_COLS）。

    R0 = encode(X)，Rp = encode(op(X))（op 施加在原始空间）；within 形式 op → rep = info(Rp) − info(R0)（变换后集合内 LOO），
    cross 形式 op → rep = info_cross(Rp → R0) − info(R0)；rcos = mean cos(R0[i], Rp[i])（polarity 除外）；rinfo = info(R0)。
    encode = 恒等展平时逐位等于 response_profile 的 ret_ / sim_ / info_raw。按 task_type 显式分发（铁律 7），现只实现分类。
    """
    validate_task_type(task_type)
    if task_type != TASK_CLASSIFICATION:
        raise NotImplementedError(f"task_type={task_type!r} 的转导型响应画像未定义（以标签一致率定义）")
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    R0 = np.asarray(encode_fn(X), float)
    base = info_loo(R0, y, chunk=chunk)
    out: dict[str, Any] = {"rinfo": float(base), "n": int(X.shape[0]), "rep_dim": int(R0.shape[1]), "seeds": [int(s) for s in seeds]}
    for op, spec in RESPONSE_OPS.items():
        use_seeds = tuple(seeds) if spec["p"] > 0 else (int(seeds[0]),)
        vals, coss = [], []
        for s in use_seeds:
            Rp = np.asarray(encode_fn(_apply_op(op, X, spec["p"], s)), float)
            if spec["form"] == "within":
                vals.append(info_loo(Rp, y, chunk=chunk) - base)
            else:
                vals.append(info_cross_rep(R0, Rp, y, chunk=chunk) - base)
            if spec["sim"] and s == use_seeds[0]:          # 余弦只取首种子（与 response_profile 的 sim_ 同口径）
                n0 = np.linalg.norm(R0, axis=1); n1 = np.linalg.norm(Rp, axis=1)
                den = n0 * n1
                coss.append(float(np.mean(np.where(den > 1e-12, (R0 * Rp).sum(1) / np.where(den > 1e-12, den, 1.0), 0.0))))
        out[f"rep_{op}"] = float(np.clip(np.mean(vals), -1.0, 1.0))
        if spec["sim"]:
            out[f"rcos_{op}"] = float(np.clip(np.mean(coss), -1.0, 1.0))
    return out
