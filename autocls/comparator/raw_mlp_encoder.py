"""S3-V 消融"w/o TS2Vec"（← AutoCTS++ §4.2.3：w/o TS2Vec uses MLP to substitute TS2Vec and generates preliminary embeddings）。

原始窗口 (n_windows, T, C) → 逐 (时间步, 通道) 共享 MLP（1 → hidden → nz）→ 通道均值（对应 Eq.10 的跨序列均值）
→ 与 TaskSetEncoder 相同的 intra / inter SetPool（Eq.11–12）→ E (nz,)。MLP 与 SetPool 都随比较器端到端训练。
论文未给 MLP 结构，按"逐时间步、无时间上下文"的语义补全（TS2Vec 给的是带上下文的逐时间步表示）。

任务无关模块（铁律 7）：只吃张量。
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from autocls.task_embed.set_transformer import SetPool


class RawMLPSetEncoder(nn.Module):
    def __init__(self, nz: int = 256, hidden: int = 64, num_heads: int = 4):
        super().__init__()
        self.nz = nz
        self.mlp = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, nz))
        self.intra_setpool = SetPool(dim_input=nz, num_outputs=1, dim_output=nz, dim_hidden=nz, num_heads=num_heads, mode='sabPF')
        self.inter_setpool = SetPool(dim_input=nz, num_outputs=1, dim_output=nz, dim_hidden=nz, num_heads=num_heads, mode='sabP')

    def set_encode(self, X: Sequence[torch.Tensor]) -> torch.Tensor:
        """一批任务（窗口数 / 长度 / 通道数可不同）→ (len(X), nz)。"""
        outs = []
        dev = next(self.parameters()).device
        for x in X:
            x = torch.as_tensor(x, dtype=torch.float32, device=dev)          # (m, T, C)
            h = self.mlp(x.unsqueeze(-1)).mean(dim=2)                          # (m, T, nz)，通道均值
            protos = self.intra_setpool(h).squeeze(1)                          # (m, nz)
            outs.append(self.inter_setpool(protos.unsqueeze(0)).squeeze(1).squeeze(0))
        return torch.stack(outs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.set_encode([x])[0]
