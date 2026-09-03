"""InfoNCE 分块的**等价性红线**与显存上界（docs/TRIPLET_OOM_FIX.md §9 的遗留项）。

MX-3-PQ 上云跑批里 12 条 OOM 跳过的轨迹有 10 条是 `loss_type=info_nce,
temporal=True`（EthanolConcentration L=1024 / SelfRegulationSCP2 L=1152）：
`resize` 把编码长度推到 n=2T≈2048~2304，temporal 的 sim 是 (B=128, n, n)，
单张 2.7 GB，而 `_masked_info_nce` 还要再物化 logits / masked_fill / log_softmax /
第二次 masked_fill 及其反向梯度各一份 —— 峰值越过 31.4 GiB 单卡。

修复与 triplet 同款（docs/TRIPLET_OOM_FIX.md §3）：**按前导（batch）维分块**，
每个实例的 InfoNCE 项彼此独立，分块求和 = 整体求和；反向手写，逐块写进同一块
梯度缓冲，故 O(chunk × n²) 在**反向**也成立（只减前向等于没修）。

本文件把**改动前的 `_masked_info_nce` 原样留作 oracle**，逐项比对：
1. 不分块路径与 oracle **逐位相同**（阈值以下零改动 → 已缓存的 R' 照样复现）；
2. 分块路径的损失值与梯度（float32/float64，最坏相对偏差 ≤ 8 eps）；
3. 分块真的分了（数块调用次数），且分块大小不改变结果；
4. 峰值显存随 chunk 而非 B 增长（GPU 上跑，含 backward）。
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import autocls.contrastive.losses as losses_mod
from autocls.contrastive.losses import (
    _masked_info_nce,
    contrastive_loss,
    cross_sim,
    masked_info_nce_from_reprs,
    masked_triplet_from_reprs,
    pairwise_sim,
)
from autocls.contrastive.pair_construction import (
    cross_scale_contrast_loss,
    hierarchical_pool,
    temporal_contrast_loss,
)


# ---------------------------------------------------------------------------
# oracle：改动前的 _masked_info_nce（losses.py 修改前版本，逐字保留）
# ---------------------------------------------------------------------------


def reference_masked_info_nce(
    sim: torch.Tensor, pos_mask: torch.Tensor, valid_mask: torch.Tensor, temperature: float
) -> torch.Tensor:
    logits = sim / temperature
    logits = logits.masked_fill(~valid_mask, float("-inf"))
    log_prob = F.log_softmax(logits, dim=-1)

    pos = pos_mask & valid_mask
    has_neg = (valid_mask & ~pos_mask).any(dim=-1, keepdim=True)
    pos = pos & has_neg
    n_pos = pos.sum()
    if n_pos == 0:
        return sim.new_tensor(0.0)
    return -(log_prob.masked_fill(~pos, 0.0).sum() / n_pos)


# ---------------------------------------------------------------------------
# 造样例
# ---------------------------------------------------------------------------


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _case(shape, *, pos_rate=0.25, seed=0, sim_func="cosine", dtype=torch.float32,
          empty_rows=()):
    """(sim, pos_mask, valid_mask)：由真实 pairwise_sim 造 sim，随机对称正样本掩码。

    empty_rows 里的行强制清空正样本（覆盖"该 anchor 无正样本，不进损失"分支）。
    掩码按 (lead, n, n) **逐 batch 不同**造，才测得出分块把哪一片给错位了。
    """
    *lead, n = shape
    z = torch.randn(*lead, n, 8, generator=_gen(seed), dtype=dtype)
    sim = pairwise_sim(z, sim_func)
    raw = torch.rand(*lead, n, n, generator=_gen(seed + 1000)) < pos_rate
    pos = raw | raw.transpose(-1, -2)
    for r in empty_rows:
        pos[..., r, :] = False
        pos[..., :, r] = False
    valid = (~torch.eye(n, dtype=torch.bool)).expand_as(sim)
    return sim, pos, valid


def _rel(got: torch.Tensor, want: torch.Tensor) -> float:
    """最坏相对偏差（以 want 的最大幅度为分母，0 张量返回 0）。"""
    scale = want.abs().max().item()
    if scale == 0.0:
        return (got - want).abs().max().item()
    return ((got - want).abs().max() / scale).item()


EPS32 = torch.finfo(torch.float32).eps
EPS64 = torch.finfo(torch.float64).eps
TOL_EPS = 8.0            # 红线：最坏相对偏差 ≤ 8 eps（对齐 triplet 的 ~2 eps 量级）

SHAPES = [(6,), (1, 9), (4, 12), (7, 8), (3, 5, 7)]
NO_CHUNK = 1 << 30       # 大到任何测试形状都走不分块路径


def _step_for(chunk_elems: int, m: int, n: int) -> int:
    return max(1, chunk_elems // max(m * n, 1))


# ---------------------------------------------------------------------------
# 1. 阈值以下逐位相同（已缓存 R' 的复现红线）
# ---------------------------------------------------------------------------


class TestUnchunkedIsBitIdentical:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("sim_func", ["dot", "cosine", "euclidean"])
    def test_value_bitwise_equal_to_reference(self, shape, sim_func):
        sim, pos, valid = _case(shape, sim_func=sim_func, seed=2)
        got = _masked_info_nce(sim, pos, valid, 0.5, chunk_elems=NO_CHUNK)
        want = reference_masked_info_nce(sim, pos, valid, 0.5)
        assert torch.equal(got, want), (got.item(), want.item())

    @pytest.mark.parametrize("shape", SHAPES)
    def test_grad_bitwise_equal_to_reference(self, shape):
        sim, pos, valid = _case(shape, seed=4)
        a = sim.clone().requires_grad_(True)
        b = sim.clone().requires_grad_(True)
        _masked_info_nce(a, pos, valid, 0.3, chunk_elems=NO_CHUNK).backward()
        reference_masked_info_nce(b, pos, valid, 0.3).backward()
        assert torch.equal(a.grad, b.grad)

    def test_default_threshold_leaves_small_inputs_unchunked(self):
        """默认阈值（2^26）下，测试规模的张量必须走原路径 → 逐位相同。"""
        sim, pos, valid = _case((4, 12), seed=6)
        got = contrastive_loss(sim, pos, loss_type="info_nce", temperature=0.4,
                               valid_mask=valid)
        want = reference_masked_info_nce(sim, pos, valid, 0.4)
        assert torch.equal(got, want)


# ---------------------------------------------------------------------------
# 2. 分块等价：损失值与梯度
# ---------------------------------------------------------------------------


class TestChunkedEquivalence:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("chunk_elems", [1, 64, 128, 200])
    def test_value_matches_reference(self, shape, chunk_elems):
        sim, pos, valid = _case(shape, seed=8)
        got = _masked_info_nce(sim, pos, valid, 0.5, chunk_elems=chunk_elems)
        want = reference_masked_info_nce(sim, pos, valid, 0.5)
        assert _rel(got, want) <= TOL_EPS * EPS32, (got.item(), want.item())

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("chunk_elems", [1, 64, 200])
    def test_grad_matches_reference(self, shape, chunk_elems):
        sim, pos, valid = _case(shape, seed=9)
        a = sim.clone().requires_grad_(True)
        b = sim.clone().requires_grad_(True)
        _masked_info_nce(a, pos, valid, 0.3, chunk_elems=chunk_elems).backward()
        reference_masked_info_nce(b, pos, valid, 0.3).backward()
        assert _rel(a.grad, b.grad) <= TOL_EPS * EPS32

    @pytest.mark.parametrize("sim_func", ["dot", "cosine", "euclidean"])
    def test_all_sim_funcs(self, sim_func):
        sim, pos, valid = _case((5, 11), sim_func=sim_func, seed=10)
        a = sim.clone().requires_grad_(True)
        b = sim.clone().requires_grad_(True)
        got = _masked_info_nce(a, pos, valid, 10.0, chunk_elems=121)
        want = reference_masked_info_nce(b, pos, valid, 10.0)
        got.backward()
        want.backward()
        assert _rel(got.detach(), want.detach()) <= TOL_EPS * EPS32
        assert _rel(a.grad, b.grad) <= TOL_EPS * EPS32

    @pytest.mark.parametrize("chunk_elems", [1, 64, 200])
    def test_float64_value_and_grad(self, chunk_elems):
        """float64 下偏差同样只有几个 eps —— 语义若变，偏差不会随精度一起缩小。"""
        sim, pos, valid = _case((5, 13), dtype=torch.float64, seed=12)
        a = sim.clone().requires_grad_(True)
        b = sim.clone().requires_grad_(True)
        got = _masked_info_nce(a, pos, valid, 0.7, chunk_elems=chunk_elems)
        want = reference_masked_info_nce(b, pos, valid, 0.7)
        got.backward()
        want.backward()
        assert _rel(got.detach(), want.detach()) <= TOL_EPS * EPS64
        assert _rel(a.grad, b.grad) <= TOL_EPS * EPS64

    def test_gradcheck_float64(self):
        """反向是手写的，必须过独立的数值梯度检查——与 oracle 一致只排除得掉
        浮点层面的问题，排不掉"公式本身抄错"。"""
        sim, pos, valid = _case((3, 7), pos_rate=0.3, dtype=torch.float64, seed=14)
        sim = sim.requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda s: _masked_info_nce(s, pos, valid, 0.9, chunk_elems=49),
            (sim,), eps=1e-6, atol=1e-8, rtol=1e-5)

    @pytest.mark.parametrize("chunk_elems", [1, 64, 200, NO_CHUNK])
    def test_chunk_size_invariant(self, chunk_elems):
        """chunk 是纯内存旋钮：任意分块大小之间互相一致。"""
        sim, pos, valid = _case((6, 10), seed=16)
        base = _masked_info_nce(sim, pos, valid, 1.0, chunk_elems=NO_CHUNK)
        got = _masked_info_nce(sim, pos, valid, 1.0, chunk_elems=chunk_elems)
        assert _rel(got, base) <= TOL_EPS * EPS32


# ---------------------------------------------------------------------------
# 3. 边界：无正样本行 / 无负样本 / 分块整除与不整除
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.parametrize("chunk_elems", [1, 100, 250])
    def test_rows_without_positive(self, chunk_elems):
        """部分 anchor 行无正样本（不进分子也不进分母）——分块不得改变归约集合。"""
        sim, pos, valid = _case((5, 10), seed=18, empty_rows=(0, 3, 7))
        a = sim.clone().requires_grad_(True)
        b = sim.clone().requires_grad_(True)
        got = _masked_info_nce(a, pos, valid, 0.5, chunk_elems=chunk_elems)
        want = reference_masked_info_nce(b, pos, valid, 0.5)
        got.backward()
        want.backward()
        assert _rel(got.detach(), want.detach()) <= TOL_EPS * EPS32
        assert _rel(a.grad, b.grad) <= TOL_EPS * EPS32
        assert (a.grad[:, 0, :] == 0).all()      # 无正样本的行梯度恒 0

    def test_batch_slice_without_any_positive(self):
        """整个 batch 分片一个正样本都没有：该片对分子分母都贡献 0，不能除爆。"""
        sim, pos, valid = _case((4, 9), seed=20)
        pos[1] = False
        pos[2] = False
        got = _masked_info_nce(sim, pos, valid, 0.5, chunk_elems=81)   # step=1
        want = reference_masked_info_nce(sim, pos, valid, 0.5)
        assert _rel(got, want) <= TOL_EPS * EPS32

    def test_no_positive_at_all_returns_zero(self):
        sim = torch.randn(3, 4, 4, generator=_gen(22))
        pos = torch.zeros(3, 4, 4, dtype=torch.bool)
        valid = (~torch.eye(4, dtype=torch.bool)).expand_as(sim)
        got = _masked_info_nce(sim, pos, valid, 1.0, chunk_elems=16)
        assert got.item() == 0.0 == reference_masked_info_nce(sim, pos, valid, 1.0).item()

    def test_no_negative_returns_zero(self):
        """全员互为正样本 → has_neg 全 False → 无可用 anchor，返回 0。"""
        sim = torch.randn(2, 3, 3, generator=_gen(24))
        pos = torch.ones(2, 3, 3, dtype=torch.bool)
        valid = (~torch.eye(3, dtype=torch.bool)).expand_as(sim)
        got = _masked_info_nce(sim, pos, valid, 1.0, chunk_elems=9)
        assert got.item() == 0.0 == reference_masked_info_nce(sim, pos, valid, 1.0).item()

    @pytest.mark.parametrize("chunk_elems", [81, 162, 200, 243, 300])
    def test_chunk_boundary_divisible_and_not(self, chunk_elems):
        """B=7 与 step ∈ {1,2,3} —— 整除与不整除都不能漏加或重复加。"""
        sim, pos, valid = _case((7, 9), pos_rate=0.35, seed=26)
        want = reference_masked_info_nce(sim, pos, valid, 1.0)
        got = _masked_info_nce(sim, pos, valid, 1.0, chunk_elems=chunk_elems)
        assert _rel(got, want) <= TOL_EPS * EPS32

    def test_two_dim_sim_has_no_batch_dim(self):
        """(n, n) 无前导维：分块退化成一块，结果仍与 oracle 一致。"""
        sim, pos, valid = _case((6,), seed=28)
        got = _masked_info_nce(sim, pos, valid, 0.5, chunk_elems=1)
        want = reference_masked_info_nce(sim, pos, valid, 0.5)
        assert _rel(got, want) <= TOL_EPS * EPS32

    def test_hand_computed_value_survives_chunking(self):
        """既有手算用例（test_contrastive.py::TestInfoNCE）在 chunk=1 下仍成立。"""
        sim = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
        pos = torch.tensor([[[False, True], [True, False]]])
        valid = (~torch.eye(2, dtype=torch.bool)).expand_as(sim)
        # 每行唯一有效候选就是正样本 → log_softmax=0；但唯一有效候选是正样本时
        # has_neg=False → 无可用 anchor → 0
        got = _masked_info_nce(sim, pos, valid, 1.0, chunk_elems=2)
        assert got.item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. 分块真的分了（不是"阈值永不触发"的假绿）
# ---------------------------------------------------------------------------


class TestChunkingActuallySplits:
    def _count_calls(self, monkeypatch):
        calls = []
        orig = losses_mod._info_nce_chunk

        def spy(s, p, v, t):
            calls.append(int(s.shape[0]))
            return orig(s, p, v, t)

        monkeypatch.setattr(losses_mod, "_info_nce_chunk", spy)
        return calls

    @pytest.mark.parametrize("chunk_elems,expect_fwd", [(81, 8), (162, 4), (324, 2)])
    def test_number_of_chunks(self, chunk_elems, expect_fwd, monkeypatch):
        """B=8、m·n=81 → step=chunk_elems//81，前向块数 = ceil(8/step)。"""
        sim, pos, valid = _case((8, 9), seed=30)
        calls = self._count_calls(monkeypatch)
        _masked_info_nce(sim, pos, valid, 0.5, chunk_elems=chunk_elems)
        assert len(calls) == expect_fwd, calls
        assert sum(calls) == 8                      # 每个 batch 元素恰好过一次

    def test_backward_is_chunked_too(self, monkeypatch):
        """反向也必须逐块重算——只减前向不减反向等于没修。"""
        sim, pos, valid = _case((8, 9), seed=31)
        s = sim.clone().requires_grad_(True)
        calls = self._count_calls(monkeypatch)
        _masked_info_nce(s, pos, valid, 0.5, chunk_elems=162).backward()
        assert len(calls) == 8                      # 前向 4 块 + 反向 4 块
        assert max(calls) == 2

    def test_below_threshold_never_calls_chunk_body(self, monkeypatch):
        sim, pos, valid = _case((8, 9), seed=32)
        calls = self._count_calls(monkeypatch)
        _masked_info_nce(sim, pos, valid, 0.5, chunk_elems=NO_CHUNK)
        assert calls == []

    def test_env_default_is_read(self):
        """模块默认值来自 AUTOCLS_INFONCE_CHUNK_ELEMS（configs/runtime.yaml 注入）。"""
        assert losses_mod.INFONCE_CHUNK_ELEMS == 1 << 26


# ---------------------------------------------------------------------------
# 5. 真实调用形状：temporal 对比 + info_nce（10 条 OOM 的爆点路径）
# ---------------------------------------------------------------------------


class TestRealPairConstruction:
    @pytest.mark.parametrize("chunk_elems", [1, 500, NO_CHUNK])
    @pytest.mark.parametrize("adj", [False, True])
    def test_temporal_contrast_matches_reference(self, chunk_elems, adj, monkeypatch):
        z1 = torch.randn(4, 9, 6, generator=_gen(41))
        z2 = torch.randn(4, 9, 6, generator=_gen(42))

        def run(elems, use_reference):
            a = z1.clone().requires_grad_(True)
            b = z2.clone().requires_grad_(True)
            with monkeypatch.context() as mp:
                mp.setattr(losses_mod, "INFONCE_CHUNK_ELEMS", elems)
                if use_reference:
                    mp.setattr(losses_mod, "_masked_info_nce",
                               lambda s, p, v, t, chunk_elems=None:
                               reference_masked_info_nce(s, p, v, t))
                loss = temporal_contrast_loss(a, b, adj_neighbor=adj, loss_type="info_nce",
                                              sim_func="cosine", temperature=0.5)
                loss.backward()
            return loss.detach(), a.grad, b.grad

        got = run(chunk_elems, False)
        want = run(NO_CHUNK, True)
        for g, w in zip(got, want):
            assert _rel(g, w) <= TOL_EPS * EPS32

    def test_instance_contrast_untouched_below_threshold(self):
        """instance 对比 (T, 2B, 2B) 在默认阈值下走不分块路径 → 逐位相同。"""
        from autocls.contrastive.pair_construction import instance_contrast_loss

        z1 = torch.randn(3, 5, 6, generator=_gen(43))
        z2 = torch.randn(3, 5, 6, generator=_gen(44))
        z = torch.cat([z1, z2], dim=0).transpose(0, 1)
        sim = pairwise_sim(z, "cosine")
        n = 6
        idx = torch.arange(3)
        pos = torch.zeros(n, n, dtype=torch.bool)
        pos[idx, idx + 3] = True
        pos[idx + 3, idx] = True
        valid = (~torch.eye(n, dtype=torch.bool)).expand_as(sim)
        want = reference_masked_info_nce(sim, pos.expand_as(sim), valid, 0.5)
        got = instance_contrast_loss(z1, z2, loss_type="info_nce", sim_func="cosine",
                                     temperature=0.5)
        assert torch.equal(got, want)


# ---------------------------------------------------------------------------
# 6. 显存上界：峰值随 chunk 而非 B（含 backward）
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 才能测显存峰值")
class TestPeakMemory:
    """合成 (B=32, n=2048, d=64)：sim 512 MiB，不分块时 logits/masked_fill/
    log_softmax/第二次 masked_fill 及其反向各再来一份（实测 +4.50× sim）。

    分块阈值取 2^23 而非默认 2^26：本机 8 GiB 只装得下 B=32，而 2^26 在 n=2048 上
    step=16 → 只分 2 块，测不出渐近行为。2^23 在这个形状上分 16 块，正好复现**上云
    真实形状**的块数（B=128, n=2304, 默认 2^26 → step=12 → 11 块）。峰值的残余
    下界是那一份与 sim 同形的梯度缓冲（1.00× sim），故上限约 4.5×。
    """

    B, N, D = 32, 2048, 64
    CHUNK = 1 << 23

    def _peak_bytes(self, chunk_elems):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.manual_seed(0)        # 两次调用必须同数据，否则等价性比对无意义
        z = torch.randn(self.B, self.N, self.D, device="cuda")
        sim = pairwise_sim(z, "cosine").requires_grad_(True)
        del z
        t = torch.arange(self.N // 2, device="cuda")
        pos = torch.zeros(self.N, self.N, dtype=torch.bool, device="cuda")
        pos[t, t + self.N // 2] = True
        pos[t + self.N // 2, t] = True
        pos = pos.expand_as(sim)
        valid = (~torch.eye(self.N, dtype=torch.bool, device="cuda")).expand_as(sim)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        loss = _masked_info_nce(sim, pos, valid, 0.5, chunk_elems=chunk_elems)
        loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        out = (peak, float(loss.detach()), sim.grad.detach().clone())
        del sim, pos, valid, loss
        torch.cuda.empty_cache()
        return out

    def test_peak_scales_with_chunk_not_batch(self):
        free, _total = torch.cuda.mem_get_info()
        if free < 4 * 2 ** 30:
            pytest.skip(f"空闲显存 {free / 2**30:.1f} GiB 不够跑不分块对照（需 ~2.8 GiB）")
        big, big_loss, big_grad = self._peak_bytes(1 << 40)
        small, small_loss, small_grad = self._peak_bytes(self.CHUNK)
        assert small < big / 3, (f"分块峰值 {small/2**20:.0f} MiB 未显著低于"
                                 f"不分块 {big/2**20:.0f} MiB")
        # 顺带在真实尺度上再验一次等价（小张量测不出大归约的结合序差异）
        assert _rel(torch.tensor(small_loss), torch.tensor(big_loss)) <= TOL_EPS * EPS32
        assert _rel(small_grad, big_grad) <= TOL_EPS * EPS32


# ===========================================================================
# Round 2：从**表征**分块 —— 全程不物化 (B, n, n) 的 sim
#
# 上云实测（2026-09-03）：round 1 部署后 EthanolConcentration_00/84631f08
# （info_nce, temporal, resize 0.5, euclidean, τ=1.0, instance+temporal, l2 norm）
# 仍以 "Tried to allocate 558 MiB … 30.12 GiB allocated by PyTorch" 挂掉，后续格
# 跑在 25.7 GiB。残余项正是 round 1 报告 §6.1 预告的那 2× sim（sim 本体 + 反向
# 梯度缓冲）：SCP2 最坏 resize 下 n=4378 → 单张 sim 9.1 GiB，两张就 18.2 GiB。
# 本节红线：temporal_contrast_loss 整调用的峰值必须小于**一整张 sim 的一半**。
# ===========================================================================


def reference_temporal_contrast_loss(z1, z2, *, adj_neighbor, sim_func, temperature):
    """改动前的 temporal_contrast_loss（info_nce 分支）：先物化 sim，再走 oracle 归约。"""
    T = z1.shape[1]
    if T == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=1)
    sim = pairwise_sim(z, sim_func)

    n = 2 * T
    t = torch.arange(T, device=z1.device)
    pos = torch.zeros(n, n, dtype=torch.bool, device=z1.device)
    pos[t, t + T] = True
    pos[t + T, t] = True
    if adj_neighbor:
        adj = torch.arange(T - 1, device=z1.device)
        for off in (0, T):
            pos[adj + off, adj + off + 1] = True
            pos[adj + off + 1, adj + off] = True
    valid = ~torch.eye(n, dtype=torch.bool, device=z1.device)
    return reference_masked_info_nce(
        sim, pos.expand_as(sim), valid.expand_as(sim), temperature)


def _temporal_pair(B, T, C, *, dtype=torch.float32, seed=0):
    z1 = torch.randn(B, T, C, generator=_gen(seed), dtype=dtype)
    z2 = torch.randn(B, T, C, generator=_gen(seed + 1), dtype=dtype)
    return z1, z2


def _run_temporal(z1, z2, *, adj, sim_func, tau, budget, monkeypatch, reference=False):
    """跑一次 temporal 对比，返回 (loss, dL/dz1, dL/dz2)。budget=None 表示不改阈值。"""
    a = z1.clone().detach().requires_grad_(True)
    b = z2.clone().detach().requires_grad_(True)
    with monkeypatch.context() as mp:
        if budget is not None:
            mp.setattr(losses_mod, "INFONCE_CHUNK_ELEMS", budget)
        if reference:
            loss = reference_temporal_contrast_loss(
                a, b, adj_neighbor=adj, sim_func=sim_func, temperature=tau)
        else:
            loss = temporal_contrast_loss(
                a, b, adj_neighbor=adj, loss_type="info_nce",
                sim_func=sim_func, temperature=tau)
        loss.backward()
    return loss.detach(), a.grad, b.grad


# --- (a) 与 oracle 等价：值 + 两个视图的梯度 ------------------------------


class TestFromReprsEquivalence:
    """B=5, T=7 → n=14, n²=196, B·n·n=980。
    budget=1 → step=1（5 块）；400 → step=2（2,2,1，**不整除**）；800 → step=4（4,1）。
    """

    B, T, C = 5, 7, 6

    @pytest.mark.parametrize("budget", [1, 400, 800])
    @pytest.mark.parametrize("sim_func", ["dot", "cosine", "euclidean"])
    @pytest.mark.parametrize("adj", [False, True])
    def test_float32_value_and_grads(self, budget, sim_func, adj, monkeypatch):
        z1, z2 = _temporal_pair(self.B, self.T, self.C, seed=50)
        got = _run_temporal(z1, z2, adj=adj, sim_func=sim_func, tau=0.5,
                            budget=budget, monkeypatch=monkeypatch)
        want = _run_temporal(z1, z2, adj=adj, sim_func=sim_func, tau=0.5,
                             budget=NO_CHUNK, monkeypatch=monkeypatch, reference=True)
        for g, w in zip(got, want):
            assert _rel(g, w) <= TOL_EPS * EPS32

    @pytest.mark.parametrize("budget", [1, 400, 800])
    @pytest.mark.parametrize("sim_func", ["dot", "cosine", "euclidean"])
    def test_float64_value_and_grads(self, budget, sim_func, monkeypatch):
        z1, z2 = _temporal_pair(self.B, self.T, self.C, dtype=torch.float64, seed=52)
        got = _run_temporal(z1, z2, adj=True, sim_func=sim_func, tau=0.7,
                            budget=budget, monkeypatch=monkeypatch)
        want = _run_temporal(z1, z2, adj=True, sim_func=sim_func, tau=0.7,
                             budget=NO_CHUNK, monkeypatch=monkeypatch, reference=True)
        for g, w in zip(got, want):
            assert _rel(g, w) <= TOL_EPS * EPS64

    def test_batch_of_one(self, monkeypatch):
        """B=1：step 至少 1，退化成单块也必须对。"""
        z1, z2 = _temporal_pair(1, 9, 5, seed=54)
        got = _run_temporal(z1, z2, adj=True, sim_func="cosine", tau=0.3,
                            budget=1, monkeypatch=monkeypatch)
        want = _run_temporal(z1, z2, adj=True, sim_func="cosine", tau=0.3,
                             budget=NO_CHUNK, monkeypatch=monkeypatch, reference=True)
        for g, w in zip(got, want):
            assert _rel(g, w) <= TOL_EPS * EPS32

    @pytest.mark.parametrize("sim_func", ["dot", "cosine", "euclidean"])
    def test_direct_entry_matches_masked_info_nce(self, sim_func):
        """masked_info_nce_from_reprs 与 pairwise_sim + _masked_info_nce 同解。"""
        z = torch.randn(4, 10, 6, generator=_gen(56))
        pos = torch.zeros(10, 10, dtype=torch.bool)
        idx = torch.arange(5)
        pos[idx, idx + 5] = True
        pos[idx + 5, idx] = True
        valid = ~torch.eye(10, dtype=torch.bool)

        a = z.clone().requires_grad_(True)
        got = masked_info_nce_from_reprs(a, sim_func, pos, valid, 0.4, chunk_elems=100)
        got.backward()

        b = z.clone().requires_grad_(True)
        sim = pairwise_sim(b, sim_func)
        want = reference_masked_info_nce(sim, pos.expand_as(sim), valid.expand_as(sim), 0.4)
        want.backward()
        assert _rel(got.detach(), want.detach()) <= TOL_EPS * EPS32
        assert _rel(a.grad, b.grad) <= TOL_EPS * EPS32


# --- (b) 手写反向的独立数值检查 -------------------------------------------


class TestFromReprsGradcheck:
    @pytest.mark.parametrize("sim_func", ["dot", "cosine", "euclidean"])
    @pytest.mark.parametrize("chunk_elems", [1, 128])
    def test_gradcheck_float64_wrt_z(self, sim_func, chunk_elems):
        """反向是「手写 ∂/∂sim + 局部 autograd 回传到 z」两段拼的，必须过数值差分。"""
        z = torch.randn(3, 8, 5, generator=_gen(58), dtype=torch.float64)
        pos = torch.zeros(8, 8, dtype=torch.bool)
        idx = torch.arange(4)
        pos[idx, idx + 4] = True
        pos[idx + 4, idx] = True
        valid = ~torch.eye(8, dtype=torch.bool)
        z = z.requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda x: masked_info_nce_from_reprs(
                x, sim_func, pos, valid, 0.9, chunk_elems=chunk_elems),
            (z,), eps=1e-6, atol=1e-8, rtol=1e-5)


# --- (c) 阈值以下逐位不变 --------------------------------------------------


class TestBelowThresholdStillBitIdentical:
    @pytest.mark.parametrize("sim_func", ["dot", "cosine", "euclidean"])
    @pytest.mark.parametrize("adj", [False, True])
    def test_temporal_bitwise_equal_to_oracle(self, sim_func, adj, monkeypatch):
        z1, z2 = _temporal_pair(5, 7, 6, seed=60)
        got = _run_temporal(z1, z2, adj=adj, sim_func=sim_func, tau=0.5,
                            budget=NO_CHUNK, monkeypatch=monkeypatch)
        want = _run_temporal(z1, z2, adj=adj, sim_func=sim_func, tau=0.5,
                             budget=NO_CHUNK, monkeypatch=monkeypatch, reference=True)
        for g, w in zip(got, want):
            assert torch.equal(g, w)

    def test_default_threshold_leaves_test_shapes_unchunked(self, monkeypatch):
        """默认阈值（2^26）下测试规模一律走原路径 —— 已缓存 R' 的复现红线。"""
        z1, z2 = _temporal_pair(5, 7, 6, seed=61)
        got = _run_temporal(z1, z2, adj=True, sim_func="cosine", tau=0.5,
                            budget=None, monkeypatch=monkeypatch)
        want = _run_temporal(z1, z2, adj=True, sim_func="cosine", tau=0.5,
                             budget=NO_CHUNK, monkeypatch=monkeypatch, reference=True)
        for g, w in zip(got, want):
            assert torch.equal(g, w)

    def test_triplet_never_takes_the_reprs_path(self, monkeypatch):
        """triplet 有自己的分块，阈值之上也必须照旧走 sim 路径。"""
        from autocls.contrastive import pair_construction as pc

        z1, z2 = _temporal_pair(5, 7, 6, seed=62)
        calls = []
        orig = pc.masked_info_nce_from_reprs

        def spy(*a, **k):
            calls.append(1)
            return orig(*a, **k)

        with monkeypatch.context() as mp:
            mp.setattr(pc, "masked_info_nce_from_reprs", spy)
            mp.setattr(losses_mod, "INFONCE_CHUNK_ELEMS", 1)
            a = z1.clone().requires_grad_(True)
            loss = temporal_contrast_loss(a, z2, adj_neighbor=True, loss_type="triplet",
                                          sim_func="cosine", temperature=0.5)
            loss.backward()
        assert calls == []
        assert torch.isfinite(a.grad).all()


# --- (d) CUDA：整调用峰值 < 半张 sim，即证明整张 sim 从未存在 --------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 才能测显存峰值")
class TestFromReprsPeakMemory:
    """(B=32, n=2048, C=64)：一整张 fp32 (B, n, n) sim = 512 MiB。红线取其一半
    （256 MiB）—— 整张 sim（或它的梯度缓冲）只要被建出来过一次，峰值必然 ≥ 512 MiB，
    这条断言就必挂。"""

    B, T, C = 32, 1024, 64          # n = 2T = 2048
    CHUNK = 1 << 22                 # step = 1：一次只活一个实例的 (n, n)

    def _peak(self, sim_func, budget):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.manual_seed(0)
        z1 = torch.randn(self.B, self.T, self.C, device="cuda", requires_grad=True)
        z2 = torch.randn(self.B, self.T, self.C, device="cuda", requires_grad=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        old = losses_mod.INFONCE_CHUNK_ELEMS
        losses_mod.INFONCE_CHUNK_ELEMS = budget
        try:
            loss = temporal_contrast_loss(z1, z2, adj_neighbor=True, loss_type="info_nce",
                                          sim_func=sim_func, temperature=0.5)
            loss.backward()
        finally:
            losses_mod.INFONCE_CHUNK_ELEMS = old
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        out = (peak, float(loss.detach()))
        del z1, z2, loss
        torch.cuda.empty_cache()
        return out

    @pytest.mark.parametrize("sim_func", ["dot", "cosine", "euclidean"])
    def test_full_sim_is_never_materialised(self, sim_func):
        full_sim = self.B * (2 * self.T) ** 2 * 4
        peak, _ = self._peak(sim_func, self.CHUNK)
        assert peak < full_sim / 2, (
            f"{sim_func}: 峰值 {peak/2**20:.0f} MiB ≥ 半张 sim "
            f"{full_sim/2/2**20:.0f} MiB —— 整张 sim 仍被物化过")


# --- (e) 端到端轨迹：新路径 vs round-1 路径 --------------------------------


class TestTrajectoryAgainstRound1:
    def test_40_iter_cpu_trajectory(self, monkeypatch, make_cls):
        """阈值之上（小 B、长 n）跑满 40 iters：新路径与 round-1 的 sim 分块路径
        逐遍损失同到 eps 量级（差异只能来自浮点结合序）。"""
        import numpy as np

        from autocls.contrastive import pair_construction as pc
        from autocls.contrastive.cl_trainer import pretrain_encoder
        from autocls.encoders.dilated_cnn import DilatedCNNEncoder

        cfg = make_cls(resize=0.2, temporal=True, adj_neighbor=True,
                       loss_type="info_nce", sim_func="cosine", temperature=0.1)
        X = np.random.default_rng(0).standard_normal((64, 96, 3)).astype(np.float32)

        def run(force_round1):
            with monkeypatch.context() as mp:
                mp.setattr(losses_mod, "INFONCE_CHUNK_ELEMS", 4096)
                if force_round1:   # 关掉表征出口 → 造 sim，再由 _masked_info_nce 分块
                    mp.setattr(pc, "should_chunk_info_nce", lambda *a, **k: False)
                torch.manual_seed(0)
                enc = DilatedCNNEncoder(input_dims=3, output_dims=16,
                                        hidden_dims=16, depth=3)
                losses = pretrain_encoder(enc, cfg, X, max_iters=40, batch_size=16,
                                          seed=0, device="cpu")
            return losses

        new_l = run(False)
        old_l = run(True)
        assert len(new_l) == len(old_l)
        worst = max(abs(a - b) / max(abs(b), 1e-12) for a, b in zip(new_l, old_l))
        assert worst <= 1e-5, f"逐遍损失相对偏差 {worst:.3e} 超出浮点结合序量级"


# ===========================================================================
# Round 2b：cross-scale 与 triplet 也走表征分块
#
# 第二轮补跑后仍剩 6 条 OOM（~30 GiB）：4 条 info_nce，其中 3 条 cross_scale=True
# （Ethanol_04/5080c0fb resize 0.3、Ethanol_04/161cc0db resize 0.9、
# SCP2_01/5fc0e960 resize 0.3），另 Ethanol_00/84631f08 无 cross-scale；
# 另 2 条 triplet（Ethanol_03/87315d0d、SCP2_00/54d10b1f，均 cross_scale=True，
# resize 0.7，euclidean / cosine）。
#
# cross_scale=True 时 total_contrastive_loss 的层级循环在**每个尺度**都会各造一次
# temporal sim (B,2T,2T) 和两次 cross-scale sim (B,Tc,Tf)，只挡最外层是不够的。
# ===========================================================================


def reference_masked_triplet(sim, pos_mask, valid_mask, temperature, margin=1.0):
    """改动前的 _masked_triplet（与 tests/test_triplet_chunking.py 的 oracle 同源，
    逐字保留：整块物化 hinge，不分块）。"""
    s = sim / temperature
    pos = pos_mask & valid_mask
    neg = valid_mask & ~pos_mask

    n, m = s.shape[-1], s.shape[-2]
    s2 = s.reshape(-1, m, n)
    pos2 = pos.reshape(-1, m, n)
    neg2 = neg.reshape(-1, m, n)

    idx = pos2.nonzero(as_tuple=False)
    if idx.shape[0] == 0:
        return sim.new_tensor(0.0)
    b_i, a_i, p_i = idx[:, 0], idx[:, 1], idx[:, 2]
    rows = s2[b_i, a_i]
    neg_rows = neg2[b_i, a_i]
    s_pos = s2[b_i, a_i, p_i]
    hinge = torch.relu(margin + rows - s_pos.unsqueeze(-1)) * neg_rows
    n_triplets = neg_rows.sum()
    if n_triplets == 0:
        return sim.new_tensor(0.0)
    return hinge.sum() / n_triplets


def _oracle_reduce(sim, pos, valid, loss_type, temperature):
    p, v = pos.expand_as(sim), valid.expand_as(sim)
    if loss_type == "info_nce":
        return reference_masked_info_nce(sim, p, v, temperature)
    return reference_masked_triplet(sim, p, v, temperature)


def reference_cross_scale_loss(z_fine, z_coarse, kernel, *, loss_type, sim_func,
                               temperature):
    """改动前的 cross_scale_contrast_loss：先物化 (B, Tc, Tf)，再走 oracle 归约。"""
    Tf, Tc = z_fine.shape[1], z_coarse.shape[1]
    sim = cross_sim(z_coarse, z_fine, sim_func)
    tc = torch.arange(Tc, device=z_fine.device)
    tf = torch.arange(Tf, device=z_fine.device)
    pos = (tf.unsqueeze(0) // kernel) == tc.unsqueeze(1)
    return _oracle_reduce(sim, pos, torch.ones_like(pos), loss_type, temperature)


def reference_temporal_triplet(z1, z2, *, adj_neighbor, sim_func, temperature):
    """改动前的 temporal_contrast_loss（triplet 分支）。"""
    T = z1.shape[1]
    z = torch.cat([z1, z2], dim=1)
    sim = pairwise_sim(z, sim_func)
    n = 2 * T
    t = torch.arange(T, device=z1.device)
    pos = torch.zeros(n, n, dtype=torch.bool, device=z1.device)
    pos[t, t + T] = True
    pos[t + T, t] = True
    if adj_neighbor:
        adj = torch.arange(T - 1, device=z1.device)
        for off in (0, T):
            pos[adj + off, adj + off + 1] = True
            pos[adj + off + 1, adj + off] = True
    valid = ~torch.eye(n, dtype=torch.bool, device=z1.device)
    return reference_masked_triplet(sim, pos.expand_as(sim), valid.expand_as(sim),
                                    temperature)


# --- triplet 的表征分块 ----------------------------------------------------


class TestTripletFromReprs:
    """B=5, T=7 → n=14, n²=196, B·n·n=980。budget 走 TRIPLET_CHUNK_ELEMS。"""

    @pytest.mark.parametrize("budget", [1, 400, 800])
    @pytest.mark.parametrize("sim_func", ["dot", "cosine", "euclidean"])
    @pytest.mark.parametrize("adj", [False, True])
    def test_temporal_matches_oracle(self, budget, sim_func, adj, monkeypatch):
        z1, z2 = _temporal_pair(5, 7, 6, seed=70)
        a, b = z1.clone().requires_grad_(True), z2.clone().requires_grad_(True)
        with monkeypatch.context() as mp:
            mp.setattr(losses_mod, "TRIPLET_CHUNK_ELEMS", budget)
            got = temporal_contrast_loss(a, b, adj_neighbor=adj, loss_type="triplet",
                                         sim_func=sim_func, temperature=0.5)
            got.backward()
        c, d = z1.clone().requires_grad_(True), z2.clone().requires_grad_(True)
        want = reference_temporal_triplet(c, d, adj_neighbor=adj, sim_func=sim_func,
                                          temperature=0.5)
        want.backward()
        assert _rel(got.detach(), want.detach()) <= TOL_EPS * EPS32
        assert _rel(a.grad, c.grad) <= TOL_EPS * EPS32
        assert _rel(b.grad, d.grad) <= TOL_EPS * EPS32

    @pytest.mark.parametrize("sim_func", ["dot", "cosine", "euclidean"])
    def test_float64(self, sim_func, monkeypatch):
        z1, z2 = _temporal_pair(5, 7, 6, dtype=torch.float64, seed=72)
        a, b = z1.clone().requires_grad_(True), z2.clone().requires_grad_(True)
        with monkeypatch.context() as mp:
            mp.setattr(losses_mod, "TRIPLET_CHUNK_ELEMS", 400)
            got = temporal_contrast_loss(a, b, adj_neighbor=True, loss_type="triplet",
                                         sim_func=sim_func, temperature=0.7)
            got.backward()
        c, d = z1.clone().requires_grad_(True), z2.clone().requires_grad_(True)
        want = reference_temporal_triplet(c, d, adj_neighbor=True, sim_func=sim_func,
                                          temperature=0.7)
        want.backward()
        assert _rel(got.detach(), want.detach()) <= TOL_EPS * EPS64
        assert _rel(a.grad, c.grad) <= TOL_EPS * EPS64

    @pytest.mark.parametrize("sim_func", ["dot", "cosine", "euclidean"])
    @pytest.mark.parametrize("chunk_elems", [1, 128])
    def test_gradcheck_float64_wrt_z(self, sim_func, chunk_elems):
        """hinge 的拐点会让数值差分失真，故整体下移 sim 让三元组远离 margin。"""
        z = torch.randn(3, 8, 5, generator=_gen(74), dtype=torch.float64) * 0.2
        pos = torch.zeros(8, 8, dtype=torch.bool)
        idx = torch.arange(4)
        pos[idx, idx + 4] = True
        pos[idx + 4, idx] = True
        valid = ~torch.eye(8, dtype=torch.bool)
        z = z.requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda x: masked_triplet_from_reprs(
                x, sim_func, pos, valid, 0.9, chunk_elems=chunk_elems),
            (z,), eps=1e-6, atol=1e-8, rtol=1e-5)

    def test_below_threshold_bitwise_identical(self, monkeypatch):
        """阈值以下必须与「没有表征出口时」的代码逐位相同。

        对照口径是**关掉出口后的 sim 路径**（round 2 之前的代码），不是 round 1
        之前的 oracle —— `_masked_triplet` 自己的 P 分块与那个 oracle 本就差 ~2 eps
        （docs/TRIPLET_OOM_FIX.md §4.1），那是上一轮已验收的差异。
        """
        from autocls.contrastive import pair_construction as pc

        z1, z2 = _temporal_pair(5, 7, 6, seed=76)

        def run(gate_on):
            a, b = z1.clone().requires_grad_(True), z2.clone().requires_grad_(True)
            with monkeypatch.context() as mp:
                mp.setattr(losses_mod, "TRIPLET_CHUNK_ELEMS", NO_CHUNK)
                if not gate_on:
                    mp.setattr(pc, "_maybe_from_reprs", lambda *a_, **k_: None)
                loss = temporal_contrast_loss(a, b, adj_neighbor=True,
                                              loss_type="triplet", sim_func="cosine",
                                              temperature=0.5)
                loss.backward()
            return loss.detach(), a.grad, b.grad

        for g, w in zip(run(True), run(False)):
            assert torch.equal(g, w)

    def test_slice_without_positive_pairs(self, monkeypatch):
        """整片 batch 没有正样本对：该片不进分子分母，梯度留 0。"""
        z = torch.randn(4, 6, 5, generator=_gen(78), requires_grad=True)
        pos = torch.zeros(4, 6, 6, dtype=torch.bool)
        pos[0, 0, 1] = pos[0, 1, 0] = True
        pos[3, 2, 3] = pos[3, 3, 2] = True
        valid = (~torch.eye(6, dtype=torch.bool)).expand(4, 6, 6)
        got = masked_triplet_from_reprs(z, "cosine", pos, valid, 0.5, chunk_elems=36)
        got.backward()
        w = torch.randn(0)  # 占位，避免 lint 报未用
        assert torch.isfinite(z.grad).all()
        assert (z.grad[1] == 0).all() and (z.grad[2] == 0).all()
        assert w.numel() == 0


# --- cross-scale 的表征分块 ------------------------------------------------


class TestCrossScaleFromReprs:
    """z_fine (B, Tf, C) / z_coarse (B, Tc, C)，sim 是**非方阵** (B, Tc, Tf)。"""

    B, TF, C, K = 4, 12, 5, 3          # Tc = 4，m·n = 48，B·m·n = 192

    def _pair(self, dtype=torch.float32, seed=80):
        zf = torch.randn(self.B, self.TF, self.C, generator=_gen(seed), dtype=dtype)
        zc = hierarchical_pool(zf, self.K, "avg")
        return zf, zc.detach()

    @pytest.mark.parametrize("budget", [1, 100, 150])
    @pytest.mark.parametrize("sim_func", ["dot", "cosine", "euclidean"])
    @pytest.mark.parametrize("loss_type", ["info_nce", "triplet"])
    def test_matches_oracle(self, budget, sim_func, loss_type, monkeypatch):
        zf, zc = self._pair()
        knob = "INFONCE_CHUNK_ELEMS" if loss_type == "info_nce" else "TRIPLET_CHUNK_ELEMS"
        a, b = zf.clone().requires_grad_(True), zc.clone().requires_grad_(True)
        with monkeypatch.context() as mp:
            mp.setattr(losses_mod, knob, budget)
            got = cross_scale_contrast_loss(a, b, self.K, loss_type=loss_type,
                                            sim_func=sim_func, temperature=0.5)
            got.backward()
        c, d = zf.clone().requires_grad_(True), zc.clone().requires_grad_(True)
        want = reference_cross_scale_loss(c, d, self.K, loss_type=loss_type,
                                          sim_func=sim_func, temperature=0.5)
        want.backward()
        assert _rel(got.detach(), want.detach()) <= TOL_EPS * EPS32
        assert _rel(a.grad, c.grad) <= TOL_EPS * EPS32      # 细尺度（候选集）
        assert _rel(b.grad, d.grad) <= TOL_EPS * EPS32      # 粗尺度（anchor 集）

    @pytest.mark.parametrize("loss_type", ["info_nce", "triplet"])
    def test_float64(self, loss_type, monkeypatch):
        zf, zc = self._pair(dtype=torch.float64, seed=82)
        knob = "INFONCE_CHUNK_ELEMS" if loss_type == "info_nce" else "TRIPLET_CHUNK_ELEMS"
        a, b = zf.clone().requires_grad_(True), zc.clone().requires_grad_(True)
        with monkeypatch.context() as mp:
            mp.setattr(losses_mod, knob, 100)
            got = cross_scale_contrast_loss(a, b, self.K, loss_type=loss_type,
                                            sim_func="euclidean", temperature=0.5)
            got.backward()
        c, d = zf.clone().requires_grad_(True), zc.clone().requires_grad_(True)
        want = reference_cross_scale_loss(c, d, self.K, loss_type=loss_type,
                                          sim_func="euclidean", temperature=0.5)
        want.backward()
        assert _rel(got.detach(), want.detach()) <= TOL_EPS * EPS64
        assert _rel(a.grad, c.grad) <= TOL_EPS * EPS64

    @pytest.mark.parametrize("loss_type", ["info_nce", "triplet"])
    def test_below_threshold_bitwise_identical(self, loss_type, monkeypatch):
        """阈值以下与「关掉表征出口」的 sim 路径逐位相同（口径同 triplet 那条）。"""
        from autocls.contrastive import pair_construction as pc

        zf, zc = self._pair(seed=84)

        def run(gate_on):
            a, b = zf.clone().requires_grad_(True), zc.clone().requires_grad_(True)
            with monkeypatch.context() as mp:
                if not gate_on:
                    mp.setattr(pc, "_maybe_from_reprs", lambda *a_, **k_: None)
                loss = cross_scale_contrast_loss(a, b, self.K, loss_type=loss_type,
                                                 sim_func="cosine", temperature=0.5)
                loss.backward()
            return loss.detach(), a.grad, b.grad

        for g, w in zip(run(True), run(False)):
            assert torch.equal(g, w)

    def test_gradcheck_float64_two_inputs(self):
        """双输入（za=粗、zb=细）两侧梯度都要过数值差分。"""
        zf = torch.randn(2, 9, 4, generator=_gen(86), dtype=torch.float64)
        zc = torch.randn(2, 3, 4, generator=_gen(87), dtype=torch.float64)
        tc, tf = torch.arange(3), torch.arange(9)
        pos = (tf.unsqueeze(0) // 3) == tc.unsqueeze(1)
        valid = torch.ones_like(pos)
        zf = zf.requires_grad_(True)
        zc = zc.requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda x, y: masked_info_nce_from_reprs(
                y, "cosine", pos, valid, 0.8, chunk_elems=1, zb=x),
            (zf, zc), eps=1e-6, atol=1e-8, rtol=1e-5)


# --- 失败格的真实形状：total_contrastive_loss（层级 + cross_scale + temporal）---


class TestFailingCellShapes:
    """复刻剩余 6 条 OOM 的 CLS 形状（缩小版），走 total_contrastive_loss 全链路：
    层级循环里每个尺度的 temporal 与 cross-scale sim 都必须被闸挡住且结果不变。"""

    @pytest.mark.parametrize("loss_type,sim_func", [
        ("info_nce", "euclidean"),      # Ethanol_00/84631f08、Ethanol_04/*
        ("info_nce", "cosine"),
        ("triplet", "euclidean"),       # Ethanol_03/87315d0d
        ("triplet", "cosine"),          # SCP2_00/54d10b1f
    ])
    @pytest.mark.parametrize("cross_scale", [True, False])
    def test_total_loss_matches_unchunked(self, loss_type, sim_func, cross_scale,
                                          monkeypatch, make_cls):
        from autocls.contrastive import pair_construction as pc
        from autocls.contrastive.pair_construction import total_contrastive_loss

        cfg = make_cls(temporal=True, adj_neighbor=True, cross_scale=cross_scale,
                       kernel=2, pool="avg", loss_type=loss_type,
                       sim_func=sim_func, temperature=0.5)
        z1 = torch.randn(4, 16, 5, generator=_gen(90))
        z2 = torch.randn(4, 16, 5, generator=_gen(91))

        def run(chunked):
            a, b = z1.clone().requires_grad_(True), z2.clone().requires_grad_(True)
            with monkeypatch.context() as mp:
                mp.setattr(losses_mod, "INFONCE_CHUNK_ELEMS", 64 if chunked else NO_CHUNK)
                mp.setattr(losses_mod, "TRIPLET_CHUNK_ELEMS", 64 if chunked else NO_CHUNK)
                if not chunked:      # 关掉表征出口 → 全程物化 sim（对照口径）
                    mp.setattr(pc, "_maybe_from_reprs", lambda *a_, **k_: None)
                loss = total_contrastive_loss(a, b, cfg)
                loss.backward()
            return loss.detach(), a.grad, b.grad

        for g, w in zip(run(True), run(False)):
            assert _rel(g, w) <= TOL_EPS * EPS32

    def test_every_scale_takes_the_gate(self, monkeypatch, make_cls):
        """cross_scale=True 时闸必须在**每个尺度**都触发一次，不能只挡最外层。"""
        from autocls.contrastive import pair_construction as pc
        from autocls.contrastive.pair_construction import total_contrastive_loss

        cfg = make_cls(temporal=True, adj_neighbor=True, cross_scale=True,
                       kernel=2, pool="avg", loss_type="info_nce",
                       sim_func="cosine", temperature=0.5)
        z1 = torch.randn(4, 16, 5, generator=_gen(92))
        z2 = torch.randn(4, 16, 5, generator=_gen(93))

        seen = []
        orig = pc._maybe_from_reprs

        def spy(za, zb, *a_, **k_):
            out = orig(za, zb, *a_, **k_)
            seen.append((za.shape[1], None if zb is None else zb.shape[1],
                         out is not None))
            return out

        with monkeypatch.context() as mp:
            mp.setattr(losses_mod, "INFONCE_CHUNK_ELEMS", 64)
            mp.setattr(pc, "_maybe_from_reprs", spy)
            total_contrastive_loss(z1, z2, cfg)   # 闸只在前向，不必回传

        # T=16 → 尺度 16/8/4/2/1。闸必须在**每个**尺度的 temporal（n=2T）与每个
        # 尺度的两次 cross-scale（粗, 细）上都被问到 —— 只挡最外层是不够的。
        temporal = [s for s in seen if s[1] is None]
        cross = [s for s in seen if s[1] is not None]
        assert [s[0] for s in temporal] == [32, 16, 8, 4]
        assert [(s[0], s[1]) for s in cross] == [
            (8, 16), (8, 16), (4, 8), (4, 8), (2, 4), (2, 4), (1, 2), (1, 2)]
        # 越阈值（B·m·n > 64）的都必须走表征路径，其余照旧走 sim
        for m, n, took in seen:
            assert took == (4 * m * (m if n is None else n) > 64), (m, n, took)


# --- CUDA：triplet / cross-scale 同样不物化整张 sim ------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 才能测显存峰值")
class TestTripletAndCrossScalePeakMemory:
    B, T, C = 32, 1024, 64
    CHUNK = 1 << 22
    # cross-scale 的 sim 是 (B, Tf/2, Tf)，比 temporal 的 (B, 2T, 2T) 小 8 倍，
    # 要让「半张 sim」这条线明显高于「块工作集 + 表征及其梯度」才测得出东西：
    # B=64, Tf=2048, C=32, step=1 → 半张 sim 256 MiB vs 实测峰值 ~85-103 MiB。
    CS_B, CS_TF, CS_C, CS_CHUNK = 64, 2048, 32, 1 << 20

    def _peak(self, fn, knob, budget):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.manual_seed(0)
        z1 = torch.randn(self.B, self.T, self.C, device="cuda", requires_grad=True)
        z2 = torch.randn(self.B, self.T, self.C, device="cuda", requires_grad=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        old = getattr(losses_mod, knob)
        setattr(losses_mod, knob, budget)
        try:
            fn(z1, z2).backward()
        finally:
            setattr(losses_mod, knob, old)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        del z1, z2
        torch.cuda.empty_cache()
        return peak

    @pytest.mark.parametrize("sim_func", ["cosine", "euclidean"])
    def test_triplet_temporal_never_builds_full_sim(self, sim_func):
        n = 2 * self.T
        full_sim = self.B * n * n * 4
        peak = self._peak(
            lambda a, b: temporal_contrast_loss(
                a, b, adj_neighbor=False, loss_type="triplet",
                sim_func=sim_func, temperature=1.0),
            "TRIPLET_CHUNK_ELEMS", self.CHUNK)
        assert peak < full_sim / 2, (
            f"triplet/{sim_func}: 峰值 {peak/2**20:.0f} MiB ≥ 半张 sim "
            f"{full_sim/2/2**20:.0f} MiB")

    @pytest.mark.parametrize("loss_type,knob", [
        ("info_nce", "INFONCE_CHUNK_ELEMS"), ("triplet", "TRIPLET_CHUNK_ELEMS")])
    def test_cross_scale_never_builds_full_sim(self, loss_type, knob):
        """粗尺度由细尺度 avg-pool（kernel=2）而来 → sim 是 (B, Tf/2, Tf)。"""
        full_sim = self.CS_B * (self.CS_TF // 2) * self.CS_TF * 4
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.manual_seed(0)
        zf = torch.randn(self.CS_B, self.CS_TF, self.CS_C, device="cuda",
                         requires_grad=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        old = getattr(losses_mod, knob)
        setattr(losses_mod, knob, self.CS_CHUNK)
        try:
            cross_scale_contrast_loss(
                zf, hierarchical_pool(zf, 2, "avg"), 2, loss_type=loss_type,
                sim_func="euclidean", temperature=1.0).backward()
        finally:
            setattr(losses_mod, knob, old)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        del zf
        torch.cuda.empty_cache()
        assert peak < full_sim / 2, (
            f"cross_scale/{loss_type}: 峰值 {peak/2**20:.0f} MiB ≥ 半张 sim "
            f"{full_sim/2/2**20:.0f} MiB")
