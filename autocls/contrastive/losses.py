"""对比损失（AutoCL 2.2 节"Contrastive Losses"；TECHNICAL_DESIGN.md 模块 3）。

组成：损失类型 {InfoNCE, Triplet} × 相似度 {Dot, Cosine, 负 Euclidean} × 温度 τ。

统一接口：先用 pairwise_sim 得候选间相似度矩阵 (..., n, n)，再由 contrastive_loss
按 pos_mask（正样本对）与 valid_mask（有效候选，排除自身/无效位置）归约。
各类对比（instance/temporal/cross-scale）只需构造各自的 sim 与掩码，见
pair_construction.py。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def cross_sim(a: torch.Tensor, b: torch.Tensor, sim_func: str) -> torch.Tensor:
    """anchor 集与候选集间的相似度。a: (..., m, d), b: (..., n, d) → (..., m, n)。

    dot：内积；cosine：单位化后内积；euclidean：负欧氏距离（AutoCL 用
    negative Euclidean distance 作相似度）。
    """
    if sim_func == "dot":
        return a @ b.transpose(-1, -2)
    if sim_func == "cosine":
        return F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).transpose(-1, -2)
    if sim_func == "euclidean":
        return -torch.cdist(a, b, p=2)
    raise ValueError(f"未知相似度函数: {sim_func}")


def pairwise_sim(z: torch.Tensor, sim_func: str) -> torch.Tensor:
    """候选集内两两相似度。z: (..., n, d) → (..., n, n)。"""
    return cross_sim(z, z, sim_func)


def _masked_info_nce(
    sim: torch.Tensor, pos_mask: torch.Tensor, valid_mask: torch.Tensor, temperature: float
) -> torch.Tensor:
    """InfoNCE：对每个含正样本的 anchor 行，-mean_pos log softmax(sim/τ)（softmax 在
    有效候选上取）。对 anchor 与正样本数取平均；无可用 anchor 时返回 0。"""
    logits = sim / temperature
    logits = logits.masked_fill(~valid_mask, float("-inf"))
    log_prob = F.log_softmax(logits, dim=-1)

    pos = pos_mask & valid_mask
    # 无负样本的 anchor 行没有可对比对象，不参与损失（与 instance 对比 B=1、
    # temporal 对比 T=1 返回 0 的语义一致）
    has_neg = (valid_mask & ~pos_mask).any(dim=-1, keepdim=True)
    pos = pos & has_neg
    n_pos = pos.sum()
    if n_pos == 0:
        return sim.new_tensor(0.0)
    return -(log_prob.masked_fill(~pos, 0.0).sum() / n_pos)


def _masked_triplet(
    sim: torch.Tensor,
    pos_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float,
    margin: float = 1.0,
) -> torch.Tensor:
    """Triplet：论文未给全（仅引 [17]），按 margin-based hinge 语义补全——
    对每个 anchor 的每对 (正, 负) 计 relu(margin + s_neg/τ - s_pos/τ)，
    对全部 (anchor, 正, 负) 三元组取平均；τ 控制相似度幅度（对齐 AutoCL
    "temperature ... controlling the magnitude of similarity scores"）。

    实现：不物化 (n × n_pos × n_neg) 三维张量；按正样本对逐行 gather，
    内存 O(#正样本对 × n)。
    """
    s = sim / temperature
    pos = pos_mask & valid_mask
    neg = valid_mask & ~pos_mask

    n = s.shape[-1]
    m = s.shape[-2]
    s2 = s.reshape(-1, m, n)
    pos2 = pos.reshape(-1, m, n)
    neg2 = neg.reshape(-1, m, n)

    idx = pos2.nonzero(as_tuple=False)  # (P, 3)：(batch, anchor, pos)
    if idx.shape[0] == 0:
        return sim.new_tensor(0.0)
    b_i, a_i, p_i = idx[:, 0], idx[:, 1], idx[:, 2]
    rows = s2[b_i, a_i]                  # (P, n)：每个正样本对所在 anchor 行
    neg_rows = neg2[b_i, a_i]            # (P, n)
    s_pos = s2[b_i, a_i, p_i]            # (P,)

    hinge = torch.relu(margin + rows - s_pos.unsqueeze(-1)) * neg_rows
    n_triplets = neg_rows.sum()
    if n_triplets == 0:
        return sim.new_tensor(0.0)
    return hinge.sum() / n_triplets


def contrastive_loss(
    sim: torch.Tensor,
    pos_mask: torch.Tensor,
    *,
    loss_type: str,
    temperature: float,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """按 loss_type 归约相似度矩阵为标量损失。

    sim/pos_mask/valid_mask 同形 (..., n, n)；valid_mask 缺省为排除对角线（自身）。
    """
    if valid_mask is None:
        n = sim.shape[-1]
        valid_mask = ~torch.eye(n, dtype=torch.bool, device=sim.device).expand_as(sim)
    if loss_type == "info_nce":
        return _masked_info_nce(sim, pos_mask, valid_mask, temperature)
    if loss_type == "triplet":
        return _masked_triplet(sim, pos_mask, valid_mask, temperature)
    raise ValueError(f"未知损失类型: {loss_type}")
