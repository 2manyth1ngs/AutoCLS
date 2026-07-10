"""轻量 Transformer encoder（TECHNICAL_DESIGN.md 模块 4，关键决策 2 的可切换分支）。

与 dilated_cnn 保持同一接口：(B, T, input_dims) → (B, T, output_dims) 逐时间步表示。
结构：Linear 投影 → 正弦位置编码 → nn.TransformerEncoder → Linear 输出头。
默认 dropout=0：与 dilated_cnn 相同的约束——掩码/噪声类扰动是 CLS 的搜索字段，
encoder 自身前向必须确定，否则比较器学到的是 encoder 噪声而非策略差异。
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _sinusoidal_position_encoding(T: int, d_model: int, device, dtype) -> torch.Tensor:
    """标准正弦位置编码 (T, d_model)，按需计算以支持任意长度。"""
    position = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(T, d_model, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: (d_model + 1) // 2])
    return pe


class TransformerEncoder(nn.Module):
    """(B, T, input_dims) → (B, T, output_dims)。"""

    def __init__(
        self,
        input_dims: int,
        output_dims: int = 320,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.d_model = d_model
        self.input_fc = nn.Linear(input_dims, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        # norm_first 与 nested-tensor 快径不兼容，显式关闭以消除 UserWarning
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.output_fc = nn.Linear(d_model, output_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, input_dims)
        nan_mask = x.isnan().any(dim=-1)  # NaN 时间步置零（与 dilated_cnn 一致）
        if nan_mask.any():
            x = x.masked_fill(nan_mask.unsqueeze(-1), 0.0)
        h = self.input_fc(x)
        h = h + _sinusoidal_position_encoding(h.shape[1], self.d_model, h.device, h.dtype)
        h = self.transformer(h)
        return self.output_fc(h)
