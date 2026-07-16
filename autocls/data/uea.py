"""UEA 多变量分类数据加载与预处理（路线 A 探针配套；不属于阶段一核心管线）。

datasets.py 是 HAR/Epilepsy 的专用加载；本模块处理 UEA 30 数据集的通用加载，
数据来自 results/uea_survey 落盘的 `data/uea_cache/<name>/<name>_{TRAIN,TEST}.ts`
（Zenodo 镜像，见 results/uea_survey/REPORT.md §0）。

与 datasets.py 的两个约定差异，本模块负责抹平：
1. aeon 返回 **通道优先** (n, C, L)；本项目约定 (N, T, C)——转置。
2. 不等长数据集（如 SpokenArabicDigits L=4-93）aeon 返回 list[(C, L_i)]——本模块
   右侧零填充到等长（填充值 = z-score 后的通道均值 0，中性）。

z-score（对齐 datasets.py 的按通道标准化，防泄漏语义）：统计量只在**真实时间步**
上计算，填充步不计入——否则短序列多的数据集统计量会被填充零拉偏。float64 统计
后转 float32（对齐 datasets.py 对 total_acc 那类小方差通道的处理）。

crop_len：长序列（Cricket L=1197、超长 EEG）按算力预算裁到定长（REPORT.md §5 的
CROP 杠杆）。裁剪取前缀（截断非重采样），保持与增强 crop 字段的时间语义一致。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "uea_cache"


def encode_labels(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """字符串/任意标签 → 连续 int64 (0..k-1) + 类目录（np.unique 升序）。

    取值仅作类 id，顺序不影响分类；保证连续以满足下游分类头/SVM 要求。
    """
    classes, inv = np.unique(np.asarray(y), return_inverse=True)
    return inv.astype(np.int64), classes


def pad_and_zscore(
    seqs: list[np.ndarray], crop_len: int | None = None, eps: float = 1e-8
) -> np.ndarray:
    """不等长序列列表 (T_i, C) → 等长 (N, T, C) float32，按通道 z-score。

    步骤：① crop_len 给定则各序列截断到前 crop_len 步；② 目标长度 = crop_len 或
    各序列真实长度最大值；③ 统计量只用截断后的真实步（填充不计入）；④ z-score 后
    写入，右侧不足处留零（= 通道均值，中性填充）。常值通道 std<eps 时按 1 处理
    （避免除零 NaN，该通道 z-score 后恒 0）。
    """
    if not seqs:
        raise ValueError("seqs 为空")
    C = seqs[0].shape[1]
    real = [np.asarray(s, dtype=np.float64)[:crop_len] if crop_len else np.asarray(s, dtype=np.float64)
            for s in seqs]
    target = crop_len if crop_len is not None else max(s.shape[0] for s in real)

    stacked = np.concatenate(real, axis=0)  # (sum_ti, C) 仅真实步
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < eps, 1.0, std)

    out = np.zeros((len(real), target, C), dtype=np.float32)
    for i, s in enumerate(real):
        t = min(s.shape[0], target)
        out[i, :t] = ((s[:t] - mean) / std).astype(np.float32)
    return out


def _to_seq_list(X) -> list[np.ndarray]:
    """aeon 加载结果 → list[(T_i, C)]（通道优先转 (T, C)）。

    aeon 等长返回 ndarray (n, C, L)，不等长返回 list/object 数组 of (C, L_i)。
    """
    if isinstance(X, np.ndarray) and X.ndim == 3:
        return [X[i].T for i in range(X.shape[0])]     # (C, L) → (L, C)
    return [np.asarray(x).T for x in X]                # 逐条 (C, L_i) → (L_i, C)


def load_uea_cached(
    name: str,
    split: str = "TRAIN",
    cache_dir: str | Path = DEFAULT_CACHE,
    crop_len: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """从 `data/uea_cache` 经 aeon 本地解析加载一个 UEA 数据集的一个切分。

    返回 X (N, T, C) float32（等长/已填充/已裁剪）、y (N,) int64（0..k-1）。
    加载不等长/含缺失一律用最宽松开关，填充与标准化由本模块统一处理（不依赖
    aeon 的 pad 行为，保证填充值与统计口径可控）。
    """
    from aeon.datasets import load_classification  # 惰性导入：核心管线不依赖 aeon

    X, y = load_classification(
        name, split=split, extract_path=str(cache_dir),
        load_equal_length=False, load_no_missing=False,
    )
    seqs = _to_seq_list(X)
    Xt = pad_and_zscore(seqs, crop_len=crop_len)
    yi, _ = encode_labels(y)
    return Xt, yi
