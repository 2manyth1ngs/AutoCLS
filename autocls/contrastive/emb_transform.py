"""嵌入变换（AutoCL 2.2 节"Embedding Transformations"；TECHNICAL_DESIGN.md 模块 3）。

作用于 encoder 输出的嵌入 z (B, T, d)：
- emb_jitter：z + N(0, p)，语义同数据端 jitter（AutoCL："Similar to the jittering
  and point masking in data augmentations"）。
- emb_mask：z ⊙ m，m[t] ~ Bernoulli(1-p) 按时间步采样。
- norm ∈ {none, layer_norm, l2}：layer_norm 在嵌入维上做（无仿射参数），
  l2 为嵌入维单位化。

施加顺序（论文未明确给出，按"增强在前、规范化在后"的通用语义补全）：
emb_jitter → emb_mask → norm。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..space.cls_space import CLS


def apply_emb_transform(
    z: torch.Tensor, cls_cfg: CLS, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """按 CLS 的嵌入变换字段处理嵌入 z (B, T, d)。

    注意：随机张量在 z 所在设备上生成时 generator 需在同设备；为与数据端增强
    共用同一 CPU generator，这里在 CPU 采样后搬运（嵌入张量通常不大）。
    """
    if cls_cfg.emb_jitter != 0.0:
        noise = torch.randn(z.shape, generator=generator, dtype=z.dtype)
        z = z + noise.to(z.device) * cls_cfg.emb_jitter
    if cls_cfg.emb_mask != 0.0:
        B, T, _ = z.shape
        keep = (torch.rand(B, T, 1, generator=generator) >= cls_cfg.emb_mask)
        z = z * keep.to(device=z.device, dtype=z.dtype)

    if cls_cfg.norm == "none":
        return z
    if cls_cfg.norm == "layer_norm":
        return F.layer_norm(z, z.shape[-1:])
    if cls_cfg.norm == "l2":
        return F.normalize(z, dim=-1)
    raise ValueError(f"未知 norm: {cls_cfg.norm}")
