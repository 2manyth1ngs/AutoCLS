"""绕过 SetTF 的任务编码器：E := TS2Vec 级均值池化向量（实验 1 诊断 (c)，
results/e_degradation_diag/PROTOCOL.md §3；DESIGN_NOTES §3.5-1(c)）。

动机：`results/task_embed_sanity/sanity_results.json` 显示 TS2Vec 级类别子集间余弦 0.65，
过（随机初始化）SetTF 后同一对 0.987——SetTF 可能把任务区分力压掉了。本编码器是
TaskSetEncoder 的零参数替身（同样暴露 `nz` 与 `set_encode(list) -> (n, nz)`），塞进 TCLSC
后其余通路（set_fc / 融合头）不变，用于隔离"SetTF 是否是 E 惰性的锅"。

可选逐维标准化（mean/std 缓冲区）：统计量由调用方只用**训练任务**拟合
（fit_standardizer），留出任务套用同一统计量——防止用留出任务统计量泄漏。

任务无关模块（铁律 7）：只吃 (n_windows, T, D) 张量。
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn


class MeanPoolTaskEncoder(nn.Module):
    """{窗口逐时间步表征} → 对窗口与时间取均值 → (可选 z-score) → E (nz=dim_input,)。"""

    def __init__(
        self,
        dim_input: int,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.nz = dim_input
        mu = torch.zeros(dim_input) if mean is None else torch.as_tensor(
            np.asarray(mean, dtype=np.float32))
        sd = torch.ones(dim_input) if std is None else torch.as_tensor(
            np.asarray(std, dtype=np.float32))
        assert mu.shape == (dim_input,) and sd.shape == (dim_input,), \
            f"标准化统计量维度须为 ({dim_input},)，得到 {tuple(mu.shape)}/{tuple(sd.shape)}"
        assert bool((sd > 0).all()), "std 必须全正（fit_standardizer 已加下限）"
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
        self.standardized = mean is not None or std is not None

    @staticmethod
    def pooled(x) -> torch.Tensor:
        """(n_windows, T, D) → (D,)：窗口与时间维均值。"""
        x = torch.as_tensor(x, dtype=torch.float32)
        assert x.ndim == 3, f"窗口表征应为 (n, T, D)，得到 {tuple(x.shape)}"
        return x.mean(dim=(0, 1))

    def set_encode(self, X: Sequence[torch.Tensor]) -> torch.Tensor:
        """一批任务（窗口数/长度可不同）→ (len(X), nz)。接口对齐 TaskSetEncoder.set_encode。"""
        outs = [(self.pooled(x).to(self.mu.device) - self.mu) / self.sd for x in X]
        return torch.stack(outs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.set_encode([x])[0]

    @staticmethod
    def fit_standardizer(
        windows: dict[str, np.ndarray], task_ids: Sequence[str], *, eps: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """用给定任务（训练任务）的池化向量拟合逐维 (mean, std)；std 下限 eps。"""
        pooled = np.stack([np.asarray(windows[t], dtype=np.float32).mean(axis=(0, 1))
                           for t in task_ids])
        mu = pooled.mean(axis=0)
        sd = pooled.std(axis=0, ddof=0).clip(min=eps)
        return mu.astype(np.float32), sd.astype(np.float32)
