"""对比损失（AutoCL 2.2 节"Contrastive Losses"；TECHNICAL_DESIGN.md 模块 3）。

组成：损失类型 {InfoNCE, Triplet} × 相似度 {Dot, Cosine, 负 Euclidean} × 温度 τ。

统一接口：先用 pairwise_sim 得候选间相似度矩阵 (..., n, n)，再由 contrastive_loss
按 pos_mask（正样本对）与 valid_mask（有效候选，排除自身/无效位置）归约。
各类对比（instance/temporal/cross-scale）只需构造各自的 sim 与掩码，见
pair_construction.py。
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

# Triplet 分块阈值：一次最多物化多少个 hinge 元素（chunk × n）。
#
# 为什么要分块（docs/TRIPLET_OOM_FIX.md，阶段1 的 69 条 OOM）：hinge 是 (P, n)，
# P = 正样本对数。SelfRegulationSCP2 L=1152 在最坏一次 resize 下 n=2T=4378、
# B=128、temporal+adj_neighbor → P≈1.7e6，单个 (P, n) float32 就 27.4 GiB，
# 前后向合计 175.9 GiB（实测外推），31.4 GiB 单卡硬装不下（与 worker 数无关）。
#
# **纯内存旋钮，不改数值语义**：分块求和 = 整体求和，只差浮点结合序；
# relu' 是 0/1 指示、三元组计数是整数，梯度逐位不受分块影响
# （等价性红线见 tests/test_triplet_chunking.py）。
# 默认 2^26 元素 = 268 MiB/张量（分块内瞬时同时活着 3~4 个）。
# 从环境变量读，好让 configs/runtime.yaml 的 env: 块统一注入到每个 worker
# （与 OMP_NUM_THREADS 同一条注入链：cloud_plan.worker_env）。
TRIPLET_CHUNK_ELEMS = int(os.environ.get("AUTOCLS_TRIPLET_CHUNK_ELEMS", 1 << 26))


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


def _pos_indices(pos2: torch.Tensor, valid2: torch.Tensor, budget: int) -> torch.Tensor:
    """(P, 3) 的正样本对下标 (batch, anchor, pos)，按前导维分片求。

    不整块 `(pos & valid).nonzero()` 的原因有二，且都与 (P, n) 无关：
    ① `pos & valid` 会物化一个与 sim 同形的 bool（SCP2 最坏 2.3 GiB）；
    ② CUDA 的 nonzero 要在**整个 numel** 上做前缀和，临时缓冲同样按 numel 计。
    分片后两项都降到 O(片 × m × n)。
    """
    nb, m, n = pos2.shape
    blk = max(1, budget // max(m * n, 1))
    parts = []
    for lo in range(0, nb, blk):
        loc = (pos2[lo:lo + blk] & valid2[lo:lo + blk]).nonzero(as_tuple=False)
        if loc.numel():
            loc[:, 0] += lo
            parts.append(loc)
    if not parts:
        return pos2.new_zeros((0, 3), dtype=torch.long)
    return parts[0] if len(parts) == 1 else torch.cat(parts)


def _chunk_rows(sim2, pos2, valid2, b_i, a_i, p_i, temperature):
    """一个分块的三件套：(rows=anchor 行/τ, s_pos, neg_rows)，形状均 (chunk, n)。

    /τ 放在 gather **之后**（改动前是先对整个 sim 除一遍再 gather）：逐元素除法，
    取值逐位相同，但省掉一个与 sim 同形的 s 张量。s_pos 从 rows 里再 gather 而不是
    第二次索引 sim2，同理省一次全尺寸操作。neg_rows 也只在块内由两个掩码现算
    （neg = valid ∧ ¬pos），不再留一个与 sim 同形的 neg。
    """
    rows = sim2[b_i, a_i] / temperature          # (chunk, n)：正样本对所在 anchor 行
    neg_rows = valid2[b_i, a_i] & ~pos2[b_i, a_i]
    return rows, rows.gather(-1, p_i.unsqueeze(-1)), neg_rows


class _TripletHingeSum(torch.autograd.Function):
    """按 P 分块的 hinge 求和 + 手写反向 —— 让 O(chunk × n) 在**反向**也成立。

    为什么不能只分块前向、反向交给 autograd（含 gradient checkpointing）：
    `sim2[b_i, a_i]` 这种高级索引的反向会分配一个与 sim2 **同形**的零张量再累加，
    每块一次；SCP2 最坏一次 resize 下 sim2 是 (128, 4378, 4378) = 9.1 GiB，
    于是"分了块也照样多占十几 GiB"（实测 32.4 GiB vs 预算 31.4）。手写反向直接
    `index_put_(accumulate=True)` 进同一块梯度缓冲，分块内就只剩 (chunk, n)。

    反向公式（h = relu(margin + r_j − r_p) · neg_j，r = sim_row/τ）：
        ∂Σh/∂r_j = m_j           j 为有效负样本
        ∂Σh/∂r_p = −Σ_j m_j      p 是正样本 → neg_p=False → m_p=0，无自身项
    其中 m = (margin + r_j − r_p > 0) ∧ neg_j —— 判据 `> 0` 与 torch.relu 的反向
    (`input > 0`) 逐位一致，故梯度与改动前实现同解（tests/test_triplet_chunking.py
    逐形状比对，另有 float64 gradcheck）。

    第二个返回值是三元组计数（不可导）：与 hinge 同在块内由 neg_rows 求得，省掉
    "先 neg.sum(-1) 再 gather" —— bool 张量整块求和会先升成 int64，(B, m, n) 一升
    就是 8 倍字节（SCP2 最坏 18.3 GiB，实测是分块后最大的残余热点）。
    """

    @staticmethod
    def forward(ctx, sim2, pos2, valid2, b_i, a_i, p_i, temperature, margin, step):
        parts, count = [], 0
        for lo in range(0, b_i.numel(), step):
            sl = slice(lo, lo + step)
            rows, s_pos, neg_rows = _chunk_rows(
                sim2, pos2, valid2, b_i[sl], a_i[sl], p_i[sl], temperature)
            parts.append((torch.relu(margin + rows - s_pos) * neg_rows).sum())
            count += int(neg_rows.sum())
        ctx.save_for_backward(sim2, pos2, valid2, b_i, a_i, p_i)
        ctx.temperature, ctx.margin, ctx.step = temperature, margin, step
        # stack + sum 而非 Python 累加：对分块部分和做树形归约，浮点精度与改动前
        # 对整块 (P, n) 调 .sum() 同量级（顺序累加会把小分块加进大累加器，掉精度）
        return torch.stack(parts).sum(), torch.tensor(count, device=sim2.device)

    @staticmethod
    def backward(ctx, grad_out, _grad_count):
        sim2, pos2, valid2, b_i, a_i, p_i = ctx.saved_tensors
        grad = torch.zeros_like(sim2)
        scale = grad_out / ctx.temperature       # r = sim/τ 的链式因子
        for lo in range(0, b_i.numel(), ctx.step):
            sl = slice(lo, lo + ctx.step)
            b, a, p = b_i[sl], a_i[sl], p_i[sl]
            rows, s_pos, neg_rows = _chunk_rows(
                sim2, pos2, valid2, b, a, p, ctx.temperature)
            m = ((ctx.margin + rows - s_pos) > 0) & neg_rows
            gr = m.to(sim2.dtype)                       # ∂/∂r_j
            gr.scatter_add_(-1, p.unsqueeze(-1), -gr.sum(-1, keepdim=True))  # ∂/∂r_p
            grad.index_put_((b, a), gr * scale, accumulate=True)
        return (grad,) + (None,) * 8


def _masked_triplet(
    sim: torch.Tensor,
    pos_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float,
    margin: float = 1.0,
    chunk_elems: Optional[int] = None,
) -> torch.Tensor:
    """Triplet：论文未给全（仅引 [17]），按 margin-based hinge 语义补全——
    对每个 anchor 的每对 (正, 负) 计 relu(margin + s_neg/τ - s_pos/τ)，
    对全部 (anchor, 正, 负) 三元组取平均；τ 控制相似度幅度（对齐 AutoCL
    "temperature ... controlling the magnitude of similarity scores"）。

    实现：不物化 (n × n_pos × n_neg) 三维张量；按正样本对逐行 gather，并把
    (P, n) 的 hinge **按 P 分块累加**，内存 O(chunk × n)（chunk_elems 缺省取
    TRIPLET_CHUNK_ELEMS）。分块求和 = 整体求和，语义不变。

    改动前的实现里，除 hinge 外还有四处与 sim 同形（而非 (P, n) 同形）的开销，
    分块后它们就成了新的峰值；一并消掉，"O(chunk × n)"才在**反向**也成立
    （只减前向等于没修 —— 阶段1 的 OOM 一半发生在 backward）：
    - 分块体是自定义 autograd.Function（_TripletHingeSum），反向手写、逐块直接
      累加进同一块梯度缓冲，不留任何 (chunk, n) 中间量给 autograd；
    - τ 归一化与 neg 掩码都移进分块体（见 _chunk_rows）：不再留与 sim 同形的
      `s = sim/τ`（及其反向梯度缓冲）和 `neg = valid ∧ ¬pos`；
    - 三元组计数在块内累加，不再对 (B, m, n) 的 bool 整块 `.sum(-1)`（会先升
      int64，8 倍字节）；
    - 正样本下标按前导维分片取（见 _pos_indices）。
    """
    n = sim.shape[-1]
    m = sim.shape[-2]
    s2 = sim.reshape(-1, m, n)
    # expand_as 保证掩码与 sim 同形（改动前靠 & 的广播），且是零拷贝视图——分块体
    # 里逐块 gather 它们，全程不物化整块掩码
    pos2 = pos_mask.expand_as(sim).reshape(-1, m, n)
    valid2 = valid_mask.expand_as(sim).reshape(-1, m, n)

    budget = int(chunk_elems if chunk_elems is not None else TRIPLET_CHUNK_ELEMS)
    idx = _pos_indices(pos2, valid2, budget)  # (P, 3)：(batch, anchor, pos)
    if idx.shape[0] == 0:
        return sim.new_tensor(0.0)
    b_i, a_i, p_i = idx[:, 0], idx[:, 1], idx[:, 2]

    step = max(1, budget // max(n, 1))
    total, n_triplets = _TripletHingeSum.apply(
        s2, pos2, valid2, b_i, a_i, p_i, temperature, margin, step)
    if n_triplets == 0:      # 无有效负样本 → 无三元组（与改动前同：返回常数 0）
        return sim.new_tensor(0.0)
    return total / n_triplets


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
