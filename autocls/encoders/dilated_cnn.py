"""TS2Vec 式 dilated CNN encoder（TECHNICAL_DESIGN.md 模块 4；AutoCL B.3）。

结构照搬 AutoCTS++ 仓库 NAS_Net/ts2vec_model/{dilated_conv.py, encoder.py}
（即 TS2Vec 官方实现）：Linear 投影 → 10 层 ConvBlock（dilation=2^i）+ 输出层
→ (B, T, 320) 逐时间步表示。默认超参照 AutoCL 附录 B.3：hidden 64、输出 320。

与 TS2Vec 原 encoder.py 的差异（有意为之，注释对应铁律 6）：
- 去掉训练期的 binomial/continuous 随机掩码——掩码类增强在 AutoCLS 中是
  CLS 的搜索字段（point_mask/emb_mask），encoder 自身必须保持确定性，
  否则比较器学到的是掩码噪声而非策略差异。
- 去掉 repr_dropout（同理，保持前向确定性）。
- 保留 NaN 时间步置零的鲁棒性处理。

"固定 encoder" 语义：架构固定；每个 CLS 各自从头预训练一份权重，不是权重冻结。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SamePadConv(nn.Module):
    """同长度输出的 dilated Conv1d（照搬 ts2vec_model/dilated_conv.py:SamePadConv）。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    """双卷积残差块（照搬 ts2vec_model/dilated_conv.py:ConvBlock）。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, final: bool = False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedCNNEncoder(nn.Module):
    """(B, T, input_dims) → (B, T, output_dims) 的逐时间步表示。"""

    def __init__(
        self,
        input_dims: int,
        output_dims: int = 320,
        hidden_dims: int = 64,
        depth: int = 10,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.depth = depth
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        channels = [hidden_dims] * depth + [output_dims]
        self.feature_extractor = nn.Sequential(*[
            ConvBlock(
                channels[i - 1] if i > 0 else hidden_dims,
                channels[i],
                kernel_size=kernel_size,
                dilation=2 ** i,
                final=(i == len(channels) - 1),
            )
            for i in range(len(channels))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, input_dims)
        nan_mask = x.isnan().any(dim=-1)  # NaN 时间步置零（对齐 ts2vec_model/encoder.py）
        if nan_mask.any():
            x = x.masked_fill(nan_mask.unsqueeze(-1), 0.0)
        x = self.input_fc(x)              # (B, T, hidden)
        x = x.transpose(1, 2)             # (B, hidden, T)
        x = self.feature_extractor(x)     # (B, output, T)
        return x.transpose(1, 2)          # (B, T, output)
