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


def _triplet_budget(chunk_elems=None) -> int:
    return int(chunk_elems if chunk_elems is not None else TRIPLET_CHUNK_ELEMS)


def should_chunk_triplet(numel: int, chunk_elems=None) -> bool:
    """相似度矩阵元素数是否越过 triplet 的分块阈值 —— 与 should_chunk_info_nce
    同义，只是走 TRIPLET_CHUNK_ELEMS 这个旋钮（hinge 侧 P 分块用的也是它）。
    """
    return numel > _triplet_budget(chunk_elems)

# InfoNCE 分块阈值：sim 的元素数（B·n·n）超过它就按前导（batch）维分块。
#
# 为什么要分块（docs/TRIPLET_OOM_FIX.md §9 明写的遗留项 —— "彻底的解法是把相似度
# 矩阵也分块（改动面覆盖 InfoNCE），本次未做"）：MX-3-PQ 上云跑批 12 条 OOM 跳过的
# 轨迹里 10 条是 `loss_type=info_nce, temporal=True`，落在 EthanolConcentration
# (L=1024) 与 SelfRegulationSCP2 (L=1152)：resize 把编码长度推到 n=2T≈2048~2304，
# temporal 的 sim 是 (B=128, n, n)，单张 2.7 GB —— 而改动前的 _masked_info_nce
# 还要再物化 logits、masked_fill、log_softmax、第二次 masked_fill 各一份，加上反向
# 的梯度缓冲：4060 实测**除 sim 外还要 4.50 × sim**（B=32/n=2048：sim 512 MiB，
# 额外 2304 MiB），外推 B=128/n=2304 即 2.7 + 12.2 = 14.9 GiB，叠上 instance 对比与
# encoder 激活就越过 31.4 GiB 单卡（阶段1 那条 SelfRegulationSCP2/a1f67db4
# info_nce n=2304 外推 21.3 GiB 是同一现象）。分块后额外只剩
# 1.00 × sim（梯度缓冲）+ 5 × 一个分块 → 同形状 6.7 GiB。
#
# **纯内存旋钮，不改数值语义**：InfoNCE 的每个 anchor 行只在自己那一片
# (m, n) 上做 log_softmax，前导维之间彼此独立，分块求和 = 整体求和，只差浮点
# 结合序（等价性红线见 tests/test_infonce_chunking.py）。
# 默认 2^26 元素 = 268 MiB/张量；阈值以下**走原路径、逐位不变**，故已缓存的 R'
# 照样复现。从环境变量读，好让 configs/runtime.yaml 的 env: 块统一注入到每个
# worker（与 AUTOCLS_TRIPLET_CHUNK_ELEMS 同一条链：cloud_plan.worker_env）。
INFONCE_CHUNK_ELEMS = int(os.environ.get("AUTOCLS_INFONCE_CHUNK_ELEMS", 1 << 26))


def _chunk_budget(chunk_elems=None) -> int:
    return int(chunk_elems if chunk_elems is not None else INFONCE_CHUNK_ELEMS)


def should_chunk_info_nce(numel: int, chunk_elems=None) -> bool:
    """相似度矩阵的元素数（B·n·n）是否越过分块阈值。

    只吃一个整数，是为了让调用点在**造 sim 之前**就问得出来 ——
    pair_construction.temporal_contrast_loss 据此决定要不要物化 (B, n, n)。
    每次调用现读模块全局，故 configs/runtime.yaml 的注入与测试里的 monkeypatch
    都即时生效。
    """
    return numel > _chunk_budget(chunk_elems)


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


def _info_nce_chunk(
    sim2: torch.Tensor, pos2: torch.Tensor, valid2: torch.Tensor, temperature: float
):
    """一个分块的 (log_prob, pos)，形状均 (chunk, m, n)。

    与不分块路径**逐行同义**（同样的 /τ → masked_fill(-inf) → log_softmax，同样的
    `pos & valid & has_neg`），只是把前导维截成一片；log_softmax 沿最后一维取，
    片与片之间没有任何耦合，所以这是精确分解而非近似。
    """
    logits = sim2 / temperature
    logits = logits.masked_fill(~valid2, float("-inf"))
    log_prob = F.log_softmax(logits, dim=-1)

    pos = pos2 & valid2
    has_neg = (valid2 & ~pos2).any(dim=-1, keepdim=True)
    return log_prob, pos & has_neg


class _ChunkedInfoNCESum(torch.autograd.Function):
    """按前导（batch）维分块的 InfoNCE 分子求和 + 手写反向 —— 让 O(chunk × m × n)
    在**反向**也成立（只减前向不减反向等于没修：阶段1 的 OOM 一半发生在 backward）。

    返回 (Σ_{(i,j)∈pos} log_prob[i,j], |pos|)；外层再做 -(Σ / |pos|)。分母是整数计数，
    与分块无关，故必须先把所有块的计数加齐再除 —— 这也是本 Function 要返回两个值
    的原因（与 _TripletHingeSum 同构）。

    为什么不用 torch.utils.checkpoint（docs/TRIPLET_OOM_FIX.md §3 的同一条经验）：
    checkpoint 的反向是"重放前向再走 autograd"，重放出来的 log_softmax/masked_fill
    仍会把整块 (chunk, m, n) 的中间量交给 autograd，且每块各留一份直到该块反向结束；
    手写反向则只留下面公式里那几个当场就地算完的张量，且梯度直接写进同一块缓冲。

    反向公式（S = Σ_i Σ_{j∈P_i} log_prob[i,j]，logits = sim/τ）：
        ∂S/∂logits[i,a] = Σ_{j∈P_i}(δ_{aj} − softmax[i,a]) = pos[i,a] − k_i·softmax[i,a]
    其中 k_i = |P_i| 是该 anchor 行的正样本数、softmax = exp(log_prob)。
    无效候选处 log_prob = −inf → softmax = 0 → 梯度 0；无正样本的行 k_i = 0 且
    pos 全 False → 梯度 0（与改动前 masked_fill 反向给 0 一致）。
    ∂/∂sim 再除 τ。等价性见 tests/test_infonce_chunking.py（含 float64 gradcheck）。
    """

    @staticmethod
    def forward(ctx, sim2, pos2, valid2, temperature, step):
        parts, n_pos = [], 0
        for lo in range(0, sim2.shape[0], step):
            sl = slice(lo, lo + step)
            log_prob, pos = _info_nce_chunk(sim2[sl], pos2[sl], valid2[sl], temperature)
            parts.append(log_prob.masked_fill(~pos, 0.0).sum())
            n_pos += int(pos.sum())
        ctx.save_for_backward(sim2, pos2, valid2)
        ctx.temperature, ctx.step = temperature, step
        # stack + sum 而非 Python 累加：对分块部分和做树形归约，浮点精度与改动前
        # 对整块 (B, m, n) 调 .sum() 同量级（顺序累加会掉精度）
        return torch.stack(parts).sum(), torch.tensor(n_pos, device=sim2.device)

    @staticmethod
    def backward(ctx, grad_out, _grad_count):
        sim2, pos2, valid2 = ctx.saved_tensors
        grad = torch.zeros_like(sim2)
        scale = grad_out / ctx.temperature          # logits = sim/τ 的链式因子
        for lo in range(0, sim2.shape[0], ctx.step):
            sl = slice(lo, lo + ctx.step)
            log_prob, pos = _info_nce_chunk(
                sim2[sl], pos2[sl], valid2[sl], ctx.temperature)
            pos_f = pos.to(sim2.dtype)
            # k 用 pos_f 求而不是对 bool 直接 .sum(-1)：bool 的维度归约会先升 int64，
            # 8 倍字节（docs/TRIPLET_OOM_FIX.md §3 ④ 实测是分块后最大的残余热点）
            k = pos_f.sum(dim=-1, keepdim=True).mul_(scale)
            pos_f.mul_(scale)       # 链式因子提前折进两个加数，省一趟全块乘法的临时量
            # out=grad[sl]：grad[sl] 是连续切片视图，直接写进梯度缓冲，块内不再多留
            # 一个 (chunk, m, n)；分块内同时活着的只有 log_prob / pos_f / exp 三张
            torch.addcmul(pos_f, log_prob.exp(), k, value=-1, out=grad[sl])
        return grad, None, None, None, None


def _chunked_info_nce(
    sim: torch.Tensor,
    pos_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float,
    budget: int,
) -> torch.Tensor:
    """_masked_info_nce 的分块实现体：语义完全相同，内存 O(chunk × m × n)。"""
    n = sim.shape[-1]
    m = sim.shape[-2]
    s2 = sim.reshape(-1, m, n)
    # expand_as 保证掩码与 sim 同形，且（掩码本就是 sim 形状或可零拷贝广播时）是视图
    pos2 = pos_mask.expand_as(sim).reshape(-1, m, n)
    valid2 = valid_mask.expand_as(sim).reshape(-1, m, n)

    step = max(1, budget // max(m * n, 1))
    total, n_pos = _ChunkedInfoNCESum.apply(s2, pos2, valid2, temperature, step)
    if n_pos == 0:      # 无可用 anchor（与改动前同：返回常数 0）
        return sim.new_tensor(0.0)
    return -(total / n_pos)


def _sim_from(za: torch.Tensor, zb, sim_func: str) -> torch.Tensor:
    """一个分块的相似度：zb 为 None 走 pairwise（temporal），否则 cross（cross-scale）。

    分块体只经由这个函数碰相似度，故 pairwise_sim/cross_sim 里加新的 sim_func 档
    不需要改任何分块代码；euclidean 的 cdist 反向也照样由局部 autograd 负责。
    """
    return pairwise_sim(za, sim_func) if zb is None else cross_sim(za, zb, sim_func)


class _ChunkedInfoNCEFromReprs(torch.autograd.Function):
    """按前导（batch）维分块的 InfoNCE —— 相似度**在块内现算现扔**，
    (B, m, n) 的 sim 从头到尾不存在。

    为什么 round 1 的 _ChunkedInfoNCESum 不够：它把 sim 当输入，于是 sim 本体与
    它的梯度缓冲这两张 (B, m, n) 拿不掉。上云实测（2026-09-03）
    EthanolConcentration_00/84631f08（info_nce + temporal + euclidean + resize 0.5
    + τ=1.0 + l2 norm）在 round 1 代码下仍以 "Tried to allocate 558 MiB …
    30.12 GiB allocated by PyTorch" 挂掉，邻格跑在 25.7 GiB —— 差的正是这 2× sim
    （SCP2 最坏 resize 下 n=2T=4378 → 单张 sim 9.1 GiB，两张 18.2 GiB）。

    本实现改吃表征：梯度缓冲降成 (B, m, C)（B=128, n=2304, C=320 → 377 MB），
    块内瞬时最大 O(step × m × n)，**与 B 和 n² 都不再成正比**（实测标度律见
    docs 与报告 Round 2 §5：峰值 ≈ 7.5 × 一个分块的字节数，与 B、n 无关）。

    forward：逐块 `_sim_from(za[b0:b1], zb[b0:b1])` → logits/log_softmax → 累加分子
    与正样本计数 → 整块丢弃（autograd.Function.forward 本就在 no-grad 下跑）。

    backward：逐块在**局部 autograd** 里重算该块的 sim（enable_grad + detach 出来的
    表征副本），用 round 1 的闭式 ∂/∂sim 造出块内 grad_sim，再一步
    `torch.autograd.grad(sim_c, (zac, zbc), grad_sim)` 回传，写进 (B, ·, C) 缓冲。

    反向公式与 _ChunkedInfoNCESum 同（S = Σ_{(i,j)∈pos} log_prob[i,j]）：
        ∂S/∂logits[i,a] = pos[i,a] − k_i·softmax[i,a]，  ∂/∂sim 再除 τ
    """

    @staticmethod
    def forward(ctx, za, zb, pos2, valid2, sim_func, temperature, step):
        parts, n_pos = [], 0
        for lo in range(0, za.shape[0], step):
            sl = slice(lo, lo + step)
            sim_c = _sim_from(za[sl], None if zb is None else zb[sl], sim_func)
            log_prob, pos = _info_nce_chunk(sim_c, pos2[sl], valid2[sl], temperature)
            parts.append(log_prob.masked_fill(~pos, 0.0).sum())
            n_pos += int(pos.sum())
        ctx.save_for_backward(za, zb, pos2, valid2)
        ctx.sim_func, ctx.temperature, ctx.step = sim_func, temperature, step
        # stack + sum：对分块部分和做树形归约，浮点精度与整块 .sum() 同量级
        return torch.stack(parts).sum(), torch.tensor(n_pos, device=za.device)

    @staticmethod
    def backward(ctx, grad_out, _grad_count):
        za, zb, pos2, valid2 = ctx.saved_tensors
        grad_a = torch.zeros_like(za)                   # (B, m, C)，全尺寸只此一份
        grad_b = None if zb is None else torch.zeros_like(zb)
        nscale = -(grad_out / ctx.temperature)          # logits = sim/τ 的链式因子取负
        for lo in range(0, za.shape[0], ctx.step):
            sl = slice(lo, lo + ctx.step)
            with torch.enable_grad():                   # 只为这一块建局部图
                zac = za[sl].detach().requires_grad_(True)
                zbc = None if zb is None else zb[sl].detach().requires_grad_(True)
                sim_c = _sim_from(zac, zbc, ctx.sim_func)
            with torch.no_grad():                       # 闭式部分不进图
                log_prob, pos = _info_nce_chunk(
                    sim_c.detach(), pos2[sl], valid2[sl], ctx.temperature)
                pos_f = pos.to(za.dtype)
                k = pos_f.sum(dim=-1, keepdim=True)
                # (pos − k·softmax)·scale：就地改写 log_prob 的存储（它没有图，
                # 改写安全），块内同时活着的只有 sim_c / log_prob / pos_f 三张
                g_sim = log_prob.exp_().mul_(k).sub_(pos_f).mul_(nscale)
            inputs = (zac,) if zbc is None else (zac, zbc)
            gs = torch.autograd.grad(sim_c, inputs, g_sim)
            grad_a[sl] = gs[0]
            if grad_b is not None:
                grad_b[sl] = gs[1]
        return grad_a, grad_b, None, None, None, None, None


class _ChunkedTripletFromReprs(torch.autograd.Function):
    """triplet 的同款表征分块（上云剩下的 2 条 OOM：Ethanol_03/87315d0d 与
    SCP2_00/54d10b1f，都是 triplet + cross_scale + resize 0.7）。

    与 InfoNCE 版的差别只有归约体：这里**不手写 ∂/∂sim**，而是在局部 autograd 里
    直接复用既有的 `_TripletHingeSum`（它本身就是按正样本对 P 分块、反向手写的
    autograd.Function，docs/TRIPLET_OOM_FIX.md §3）。于是 hinge 侧的 O(chunk × n)
    与 sim 侧的 O(step × m × n) 两层分块叠在一起，且**判据/公式一字未改**——
    `_masked_triplet` 的语义由它自己的 85 个既有用例守着。

    归约仍是"先把各块的 hinge 和与三元组计数加齐，再整体相除"，故与不分块的
    `hinge.sum() / n_triplets` 精确同解（只差浮点结合序）。
    """

    @staticmethod
    def forward(ctx, za, zb, pos2, valid2, sim_func, temperature, margin, step, budget):
        parts, count = [], 0
        for lo in range(0, za.shape[0], step):
            sl = slice(lo, lo + step)
            p_c, v_c = pos2[sl], valid2[sl]
            idx = _pos_indices(p_c, v_c, budget)
            if idx.shape[0] == 0:
                continue
            sim_c = _sim_from(za[sl], None if zb is None else zb[sl], sim_func)
            step_p = max(1, budget // max(sim_c.shape[-1], 1))
            total_c, cnt_c = _TripletHingeSum.apply(
                sim_c, p_c, v_c, idx[:, 0], idx[:, 1], idx[:, 2],
                temperature, margin, step_p)
            parts.append(total_c)
            count += int(cnt_c)
        ctx.save_for_backward(za, zb, pos2, valid2)
        ctx.sim_func, ctx.temperature, ctx.margin = sim_func, temperature, margin
        ctx.step, ctx.budget = step, budget
        if not parts:
            return za.new_tensor(0.0), torch.tensor(0, device=za.device)
        return torch.stack(parts).sum(), torch.tensor(count, device=za.device)

    @staticmethod
    def backward(ctx, grad_out, _grad_count):
        za, zb, pos2, valid2 = ctx.saved_tensors
        grad_a = torch.zeros_like(za)
        grad_b = None if zb is None else torch.zeros_like(zb)
        for lo in range(0, za.shape[0], ctx.step):
            sl = slice(lo, lo + ctx.step)
            p_c, v_c = pos2[sl], valid2[sl]
            idx = _pos_indices(p_c, v_c, ctx.budget)
            if idx.shape[0] == 0:
                continue        # 该片无正样本对 → 对分子分母都不贡献，梯度留 0
            with torch.enable_grad():
                zac = za[sl].detach().requires_grad_(True)
                zbc = None if zb is None else zb[sl].detach().requires_grad_(True)
                sim_c = _sim_from(zac, zbc, ctx.sim_func)
                step_p = max(1, ctx.budget // max(sim_c.shape[-1], 1))
                total_c, _cnt = _TripletHingeSum.apply(
                    sim_c, p_c, v_c, idx[:, 0], idx[:, 1], idx[:, 2],
                    ctx.temperature, ctx.margin, step_p)
            inputs = (zac,) if zbc is None else (zac, zbc)
            gs = torch.autograd.grad(total_c, inputs, grad_out)
            grad_a[sl] = gs[0]
            if grad_b is not None:
                grad_b[sl] = gs[1]
        return (grad_a, grad_b) + (None,) * 7


def _reprs_masks(za, zb, pos_mask, valid_mask):
    """把掩码零拷贝广播到 (B, m, n)，并返回 (B, m, n, pos2, valid2)。

    broadcast_to 出来的是 stride-0 视图，逐块切片仍是视图 —— 整块掩码不物化
    （temporal/cross-scale 传进来的都是两视图共用的 (m, n)）。
    """
    B, m = za.shape[0], za.shape[1]
    n = m if zb is None else zb.shape[1]
    return (B, m, n,
            torch.broadcast_to(pos_mask, (B, m, n)),
            torch.broadcast_to(valid_mask, (B, m, n)))


def masked_info_nce_from_reprs(
    za: torch.Tensor,
    sim_func: str,
    pos_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float,
    chunk_elems: Optional[int] = None,
    zb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """InfoNCE，输入是**表征**而不是相似度矩阵。

    语义与 `_masked_info_nce(_sim_from(za, zb, sim_func), pos, valid, τ)` 逐条相同
    （无正样本的 anchor 行不进损失、分母是全局正样本对计数、掩码用法一致），
    区别只在 (B, m, n) 的 sim 全程不物化：峰值 O(step × m × n) + 一份 (B, ·, C)。

    za: (B, m, C)；zb=None 表示 pairwise（temporal 的 (B, 2T, C) 自比），否则
    za 是 anchor 集、zb 是候选集（cross-scale 的 (粗, 细)）。掩码需可广播到
    (B, m, n)。step = 分块预算 // (m·n)（至少 1）。

    调用点判"要不要走这条"用 should_chunk_info_nce（只吃元素数，好让调用点在
    **造 sim 之前**就问得出来）。阈值以下一行都不走这里，逐位与改动前相同。
    """
    B, m, n, pos2, valid2 = _reprs_masks(za, zb, pos_mask, valid_mask)
    step = max(1, _chunk_budget(chunk_elems) // max(m * n, 1))
    total, n_pos = _ChunkedInfoNCEFromReprs.apply(
        za, zb, pos2, valid2, sim_func, temperature, step)
    if n_pos == 0:      # 无可用 anchor（与 _masked_info_nce 同：返回常数 0）
        return za.new_tensor(0.0)
    return -(total / n_pos)


def masked_triplet_from_reprs(
    za: torch.Tensor,
    sim_func: str,
    pos_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float,
    margin: float = 1.0,
    chunk_elems: Optional[int] = None,
    zb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Triplet，输入是**表征**而不是相似度矩阵。

    语义与 `_masked_triplet(_sim_from(za, zb, sim_func), pos, valid, τ, margin)`
    相同（同一个 hinge、同一个"对全部三元组取平均"），区别只在 sim 逐块现算现扔。
    分块预算复用 TRIPLET_CHUNK_ELEMS（与 hinge 侧的 P 分块同一个旋钮、同一条注入链）。
    """
    B, m, n, pos2, valid2 = _reprs_masks(za, zb, pos_mask, valid_mask)
    budget = _triplet_budget(chunk_elems)
    step = max(1, budget // max(m * n, 1))
    total, count = _ChunkedTripletFromReprs.apply(
        za, zb, pos2, valid2, sim_func, temperature, margin, step, budget)
    if count == 0:      # 无有效三元组（与 _masked_triplet 同：返回常数 0）
        return za.new_tensor(0.0)
    return total / count


def _masked_info_nce(
    sim: torch.Tensor, pos_mask: torch.Tensor, valid_mask: torch.Tensor, temperature: float,
    chunk_elems: Optional[int] = None,
) -> torch.Tensor:
    """InfoNCE：对每个含正样本的 anchor 行，-mean_pos log softmax(sim/τ)（softmax 在
    有效候选上取）。对 anchor 与正样本数取平均；无可用 anchor 时返回 0。

    sim 元素数超过 chunk_elems（缺省 INFONCE_CHUNK_ELEMS）时改走**按前导维分块**的
    实现（_chunked_info_nce）：除 sim 外的开销从 4.50 × sim 降到 1.00 × sim（梯度
    缓冲）+ O(chunk × n²)，前后向都算。阈值以下**一行代码都不走新路径**，逐位与改动前相同 ——
    已缓存的 R' 的复现红线（tests/test_infonce_chunking.py::TestUnchunkedIsBitIdentical）。
    """
    budget = _chunk_budget(chunk_elems)
    if should_chunk_info_nce(sim.numel(), budget):
        return _chunked_info_nce(sim, pos_mask, valid_mask, temperature, budget)

    # ↓↓↓ 以下为改动前的实现，逐字保留（阈值以下零改动） ↓↓↓
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

    budget = _triplet_budget(chunk_elems)
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
