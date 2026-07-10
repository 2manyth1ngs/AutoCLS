"""对比对构造（AutoCL 2.2 节"Contrastive Pair Construction"；TECHNICAL_DESIGN.md 模块 3）。

三类对比 + 层级 pooling：
- instance：跨视图同实例 (z1[i], z2[i]) 为正，批内异实例为负（恒开启）。
- temporal：同实例内跨视图同时间步为正；adj_neighbor=True 时把同视图时间相邻
  嵌入 h[t±1] 也纳入正样本（AutoCL："optionally also h1[t] and its temporally
  adjacent embeddings h1[t−1], h1[t+1]"）。
- cross-scale：相邻尺度间，粗尺度位置与其池化来源的细尺度位置为正、其余为负
  （AutoCL："if h(s+1)[t_{s+1}] is pooled from h(s)[t_s], then they should be
  pulled closer, otherwise, should be pushed away"）。
- 层级 pooling：kernel ∈ {2,3,5}（stride=kernel）反复池化得多尺度嵌入，
  instance/temporal 对比施加于所有尺度（AutoCL："applied for all the scales"，
  对齐 TS2Vec losses.py 的层级归约）；kernel=0 或无效档 None 表示单尺度。

嵌入约定 (B, T, C)：B 实例数、T 时间步、C 嵌入维。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..space.cls_space import CLS
from .losses import contrastive_loss, cross_sim, pairwise_sim


def instance_contrast_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    *,
    loss_type: str,
    sim_func: str,
    temperature: float,
) -> torch.Tensor:
    """instance 对比：每个时间步上，2B 个候选中 (i, B+i) 互为正样本。

    构造对齐 TS2Vec losses.py instance_contrastive_loss（cat 两视图 → 按时间步
    做 2B×2B 相似度），但相似度/损失/温度由 CLS 字段决定。B=1 时无负样本，返回 0。
    """
    B = z1.shape[0]
    if B == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=0)      # (2B, T, C)
    z = z.transpose(0, 1)               # (T, 2B, C)：每个时间步一组候选
    sim = pairwise_sim(z, sim_func)     # (T, 2B, 2B)

    n = 2 * B
    idx = torch.arange(B, device=z1.device)
    pos = torch.zeros(n, n, dtype=torch.bool, device=z1.device)
    pos[idx, idx + B] = True
    pos[idx + B, idx] = True
    valid = ~torch.eye(n, dtype=torch.bool, device=z1.device)

    return contrastive_loss(
        sim, pos.expand_as(sim), loss_type=loss_type,
        temperature=temperature, valid_mask=valid.expand_as(sim),
    )


def temporal_contrast_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    *,
    adj_neighbor: bool,
    loss_type: str,
    sim_func: str,
    temperature: float,
) -> torch.Tensor:
    """temporal 对比：每个实例内，2T 个候选中跨视图同时间步 (t, T+t) 互为正。

    构造对齐 TS2Vec losses.py temporal_contrastive_loss；adj_neighbor=True 时
    追加同视图相邻时间步 (t, t±1) 为正样本。T=1 时无负样本，返回 0。
    """
    T = z1.shape[1]
    if T == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=1)      # (B, 2T, C)：每个实例一组候选
    sim = pairwise_sim(z, sim_func)     # (B, 2T, 2T)

    n = 2 * T
    t = torch.arange(T, device=z1.device)
    pos = torch.zeros(n, n, dtype=torch.bool, device=z1.device)
    pos[t, t + T] = True
    pos[t + T, t] = True
    if adj_neighbor:
        adj = torch.arange(T - 1, device=z1.device)
        for off in (0, T):  # 两个视图各自的同视图相邻对
            pos[adj + off, adj + off + 1] = True
            pos[adj + off + 1, adj + off] = True
    valid = ~torch.eye(n, dtype=torch.bool, device=z1.device)

    return contrastive_loss(
        sim, pos.expand_as(sim), loss_type=loss_type,
        temperature=temperature, valid_mask=valid.expand_as(sim),
    )


def hierarchical_pool(z: torch.Tensor, kernel: int, pool: str) -> torch.Tensor:
    """(B, T, C) → (B, ceil(T/kernel), C)：kernel=stride 的时间维池化。

    ceil_mode=True 让末段不足 kernel 的时间步也保留（否则短序列信息被丢弃）。
    """
    zt = z.transpose(1, 2)  # (B, C, T)
    if pool == "avg":
        out = F.avg_pool1d(zt, kernel_size=kernel, stride=kernel, ceil_mode=True)
    elif pool == "max":
        out = F.max_pool1d(zt, kernel_size=kernel, stride=kernel, ceil_mode=True)
    else:
        raise ValueError(f"未知 pooling: {pool}")
    return out.transpose(1, 2)


def cross_scale_contrast_loss(
    z_fine: torch.Tensor,
    z_coarse: torch.Tensor,
    kernel: int,
    *,
    loss_type: str,
    sim_func: str,
    temperature: float,
) -> torch.Tensor:
    """cross-scale 对比：粗尺度位置 t 与其池化来源细尺度位置
    [t·k, min((t+1)·k, T_fine)) 互为正样本，同实例其余细尺度位置为负。

    论文只给出"来源位置拉近、其余推远"的语义（2.2 节），anchor 方向未指明，
    此处按语义补全：以粗尺度为 anchor、细尺度为候选集（单向）。
    """
    B, Tf, _ = z_fine.shape
    Tc = z_coarse.shape[1]
    sim = cross_sim(z_coarse, z_fine, sim_func)  # (B, Tc, Tf)

    tc = torch.arange(Tc, device=z_fine.device)
    tf = torch.arange(Tf, device=z_fine.device)
    # pos[t, j] = (j // kernel == t)
    pos = (tf.unsqueeze(0) // kernel) == tc.unsqueeze(1)  # (Tc, Tf)
    valid = torch.ones_like(pos)

    return contrastive_loss(
        sim, pos.expand_as(sim), loss_type=loss_type,
        temperature=temperature, valid_mask=valid.expand_as(sim),
    )


def total_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, cls_cfg: CLS) -> torch.Tensor:
    """按 CLS 配置组合全部启用的对比维度，对各尺度、各维度的损失取平均
    （对齐 TS2Vec hierarchical_contrastive_loss 的 loss/d 归约）。

    kernel 为 None（无效档）或 0 时单尺度；否则反复池化直到 T=1，每个尺度都施加
    instance（恒开启）与 temporal（若开启）；cross_scale（若开启）作用于相邻尺度。
    """
    kw = dict(loss_type=cls_cfg.loss_type, sim_func=cls_cfg.sim_func,
              temperature=cls_cfg.temperature)
    multi_scale = cls_cfg.kernel not in (None, 0)

    losses: list[torch.Tensor] = []
    while True:
        losses.append(instance_contrast_loss(z1, z2, **kw))
        if cls_cfg.temporal and z1.shape[1] > 1:
            losses.append(
                temporal_contrast_loss(z1, z2, adj_neighbor=bool(cls_cfg.adj_neighbor), **kw)
            )
        if not multi_scale or z1.shape[1] <= 1:
            break
        p1 = hierarchical_pool(z1, cls_cfg.kernel, cls_cfg.pool)
        p2 = hierarchical_pool(z2, cls_cfg.kernel, cls_cfg.pool)
        if cls_cfg.cross_scale:
            losses.append(cross_scale_contrast_loss(z1, p1, cls_cfg.kernel, **kw))
            losses.append(cross_scale_contrast_loss(z2, p2, cls_cfg.kernel, **kw))
        z1, z2 = p1, p2

    return torch.stack(losses).mean()
