"""Triplet 分块求和的**等价性红线**与内存上界（docs/STAGE1_REPORT.md ③ 的修复）。

阶段1 有 69 条 (task, cls) 因显存跑不了，全部集中在 triplet × 大 crop × L≥512：
`_masked_triplet` 原先把 hinge=(P, n) 整个物化，SCP2（L=1152, B=128）单个 (P,n)
float32 就 7.1 GiB，前后向合计 ~30 GiB，31.4 GiB 单卡装不下。

修复是**纯内存优化**：按 P 分块累加求和（Σ_chunk = Σ_全体）。语义一字不改是红线，
故本文件把**改动前的实现原样留作 oracle**，逐项比对：
1. 损失值（分块累加只有浮点结合序差异）；
2. **对 sim 的梯度**（真正驱动训练的量；relu' 是 0/1 指示，分块不改变它）；
3. 分块大小不改变结果（chunk=1 与 chunk≥P 同解）；
4. 峰值显存随 chunk 而非 P 增长（GPU 上跑，含 backward——只减前向不减反向等于没修）。
"""
from __future__ import annotations

import pytest
import torch

from autocls.contrastive.losses import _masked_triplet, contrastive_loss, pairwise_sim
from autocls.contrastive.pair_construction import temporal_contrast_loss


# ---------------------------------------------------------------------------
# oracle：改动前的 _masked_triplet（losses.py 修改前版本，逐字保留）
# ---------------------------------------------------------------------------


def reference_masked_triplet(
    sim: torch.Tensor,
    pos_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float,
    margin: float = 1.0,
) -> torch.Tensor:
    s = sim / temperature
    pos = pos_mask & valid_mask
    neg = valid_mask & ~pos_mask

    n = s.shape[-1]
    m = s.shape[-2]
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


# ---------------------------------------------------------------------------
# 造样例：覆盖单/多 batch、稀疏/稠密正样本、有无负样本
# ---------------------------------------------------------------------------


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _case(shape, *, pos_rate=0.25, seed=0, sim_func="cosine", dtype=torch.float32):
    """(sim, pos_mask, valid_mask)：由真实 pairwise_sim 造 sim，随机对称正样本掩码。"""
    *lead, n = shape
    z = torch.randn(*lead, n, 8, generator=_gen(seed), dtype=dtype)
    sim = pairwise_sim(z, sim_func)
    g = _gen(seed + 1000)
    raw = torch.rand(n, n, generator=g) < pos_rate
    pos = (raw | raw.T).expand_as(sim)
    valid = (~torch.eye(n, dtype=torch.bool)).expand_as(sim)
    return sim, pos, valid


SHAPES = [(6,), (1, 9), (4, 12), (3, 5, 7)]
CHUNKS = [1, 7, 64, 1 << 20]   # 1 = 每个正样本对一块；1<<20 = 实际不分块


# ---------------------------------------------------------------------------
# 1. 损失值等价
# ---------------------------------------------------------------------------


class TestValueEquivalence:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("chunk", CHUNKS)
    def test_matches_reference(self, shape, chunk):
        sim, pos, valid = _case(shape)
        got = _masked_triplet(sim, pos, valid, 0.5, chunk_elems=chunk)
        want = reference_masked_triplet(sim, pos, valid, 0.5)
        assert torch.allclose(got, want, rtol=1e-6, atol=1e-6), (got.item(), want.item())

    @pytest.mark.parametrize("sim_func", ["dot", "cosine", "euclidean"])
    def test_all_sim_funcs(self, sim_func):
        sim, pos, valid = _case((4, 11), sim_func=sim_func, seed=3)
        got = _masked_triplet(sim, pos, valid, 10.0, chunk_elems=3)
        want = reference_masked_triplet(sim, pos, valid, 10.0)
        assert torch.allclose(got, want, rtol=1e-6, atol=1e-6)

    def test_float64_is_bitwise_close(self):
        """float64 下分块累加的偏差应落在 1e-14 量级（证明差异纯粹来自结合序）。"""
        sim, pos, valid = _case((3, 16), dtype=torch.float64, seed=7)
        got = _masked_triplet(sim, pos, valid, 0.7, chunk_elems=2)
        want = reference_masked_triplet(sim, pos, valid, 0.7)
        assert abs(got.item() - want.item()) < 1e-13

    @pytest.mark.parametrize("chunk", CHUNKS)
    def test_chunk_size_invariant(self, chunk):
        """不同 chunk 之间互相一致（chunk 是纯内存旋钮，不得改变结果）。"""
        sim, pos, valid = _case((5, 13), seed=11)
        base = _masked_triplet(sim, pos, valid, 1.0, chunk_elems=1 << 30)
        got = _masked_triplet(sim, pos, valid, 1.0, chunk_elems=chunk)
        assert torch.allclose(got, base, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. 梯度等价（训练真正用到的量）
# ---------------------------------------------------------------------------


class TestGradientEquivalence:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("chunk", [1, 5, 1 << 20])
    def test_grad_wrt_sim_matches(self, shape, chunk):
        sim, pos, valid = _case(shape, seed=5)
        a = sim.clone().requires_grad_(True)
        b = sim.clone().requires_grad_(True)
        _masked_triplet(a, pos, valid, 0.3, chunk_elems=chunk).backward()
        reference_masked_triplet(b, pos, valid, 0.3).backward()
        assert a.grad is not None and b.grad is not None
        assert torch.allclose(a.grad, b.grad, rtol=1e-6, atol=1e-7)

    def test_grad_wrt_embedding_matches(self):
        """端到端一点：梯度要能穿过 pairwise_sim 回到嵌入，且与 oracle 一致。"""
        z0 = torch.randn(4, 10, 8, generator=_gen(2))
        pos = (torch.rand(10, 10, generator=_gen(9)) < 0.3)
        pos = (pos | pos.T).expand(4, 10, 10)
        valid = (~torch.eye(10, dtype=torch.bool)).expand(4, 10, 10)

        grads = []
        for fn in (lambda s: _masked_triplet(s, pos, valid, 0.5, chunk_elems=4),
                   lambda s: reference_masked_triplet(s, pos, valid, 0.5)):
            z = z0.clone().requires_grad_(True)
            fn(pairwise_sim(z, "cosine")).backward()
            grads.append(z.grad)
        assert torch.allclose(grads[0], grads[1], rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize("chunk", [1, 5, 1 << 20])
    def test_gradcheck_float64(self, chunk):
        """反向是手写的（_TripletHingeSum.backward），必须过独立的数值梯度检查——
        与 oracle 一致只能证明"两边同错"不会发生在浮点层面，证明不了公式本身对。"""
        sim, pos, valid = _case((2, 7), pos_rate=0.35, seed=41, dtype=torch.float64)
        # margin 附近的 hinge 拐点会让数值差分失真：整体下移 sim 使各三元组远离 0
        sim = (sim - 3.0).requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda s: _masked_triplet(s, pos, valid, 0.9, chunk_elems=chunk),
            (sim,), eps=1e-6, atol=1e-8, rtol=1e-5)

    def test_grad_finite_under_tiny_chunk(self):
        sim, pos, valid = _case((2, 20), seed=13)
        s = sim.clone().requires_grad_(True)
        _masked_triplet(s, pos, valid, 0.1, chunk_elems=1).backward()
        assert torch.isfinite(s.grad).all()


# ---------------------------------------------------------------------------
# 3. 边界：无正样本 / 无负样本 / 分块整除与不整除
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_positive_returns_zero(self):
        sim = torch.eye(4)
        pos = torch.zeros(4, 4, dtype=torch.bool)
        valid = ~torch.eye(4, dtype=torch.bool)
        assert _masked_triplet(sim, pos, valid, 1.0, chunk_elems=1).item() == 0.0

    def test_no_negative_returns_zero(self):
        """全员互为正样本 → 无三元组，返回 0（与 oracle 同）。"""
        sim = torch.randn(3, 3, generator=_gen())
        pos = torch.ones(3, 3, dtype=torch.bool)
        valid = ~torch.eye(3, dtype=torch.bool)
        got = _masked_triplet(sim, pos, valid, 1.0, chunk_elems=1)
        assert got.item() == 0.0 == reference_masked_triplet(sim, pos, valid, 1.0).item()

    @pytest.mark.parametrize("chunk_elems", [8, 9, 10, 16, 17])
    def test_chunk_boundary_divisible_and_not(self, chunk_elems):
        """chunk 恰好整除 / 不整除 P 都不能漏加或重复加。"""
        sim, pos, valid = _case((1, 8), pos_rate=0.5, seed=17)
        got = _masked_triplet(sim, pos, valid, 1.0, chunk_elems=chunk_elems)
        want = reference_masked_triplet(sim, pos, valid, 1.0)
        assert torch.allclose(got, want, rtol=1e-6, atol=1e-6)

    def test_hand_computed_value_survives_chunking(self):
        """既有手算用例（test_contrastive.py::TestTriplet）在 chunk=1 下仍成立。"""
        sim = torch.tensor([[9.0, 2.0, 3.0], [2.0, 9.0, 4.0], [3.0, 4.0, 9.0]])
        pos = torch.zeros(3, 3, dtype=torch.bool)
        pos[0, 1] = pos[1, 0] = True
        valid = ~torch.eye(3, dtype=torch.bool)
        got = _masked_triplet(sim, pos, valid, 1.0, chunk_elems=1)
        assert got.item() == pytest.approx(2.5)

    def test_default_chunk_used_when_omitted(self):
        """不传 chunk_elems 时走模块默认值，结果仍与 oracle 一致。"""
        sim, pos, valid = _case((4, 12), seed=23)
        got = contrastive_loss(sim, pos, loss_type="triplet", temperature=0.4,
                               valid_mask=valid)
        want = reference_masked_triplet(sim, pos, valid, 0.4)
        assert torch.allclose(got, want, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. 真实调用形状：temporal 对比 + adj_neighbor（69 条 OOM 的爆点路径）
# ---------------------------------------------------------------------------


class TestRealPairConstruction:
    @pytest.mark.parametrize("chunk", [1, 33, 1 << 20])
    def test_temporal_contrast_matches_reference(self, chunk, monkeypatch):
        """temporal 对比（adj_neighbor=True）下，分块与 oracle 同值同梯度。"""
        import autocls.contrastive.losses as losses_mod

        z1 = torch.randn(3, 9, 6, generator=_gen(31), requires_grad=True)
        z2 = torch.randn(3, 9, 6, generator=_gen(32), requires_grad=True)

        def run(chunk_elems, use_reference):
            a, b = z1.clone().detach().requires_grad_(True), z2.clone().detach().requires_grad_(True)
            with monkeypatch.context() as mp:
                mp.setattr(losses_mod, "TRIPLET_CHUNK_ELEMS", chunk_elems)
                if use_reference:
                    mp.setattr(losses_mod, "_masked_triplet",
                               lambda s, p, v, t, margin=1.0, chunk_elems=None:
                               reference_masked_triplet(s, p, v, t, margin))
                loss = temporal_contrast_loss(a, b, adj_neighbor=True, loss_type="triplet",
                                              sim_func="cosine", temperature=0.5)
                loss.backward()
            return loss.detach(), a.grad, b.grad

        got_l, got_g1, got_g2 = run(chunk, False)
        want_l, want_g1, want_g2 = run(1 << 30, True)
        assert torch.allclose(got_l, want_l, rtol=1e-6, atol=1e-6)
        assert torch.allclose(got_g1, want_g1, rtol=1e-5, atol=1e-7)
        assert torch.allclose(got_g2, want_g2, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# 5. 内存上界：峰值随 chunk 而非 P（含 backward）
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 才能测显存峰值")
class TestPeakMemory:
    """每 anchor 多个正样本（真实 temporal+adj_neighbor 就是 ~3 个）才让 (P, n) 压过
    sim 的 (B, n, n)——P = k·B·n 时 P·n = k 倍 sim。k=1 的形状测不出任何东西。"""

    B, N, K = 2, 1024, 12          # k≈12 → (P, n) 约 12 倍 sim 大小

    def _peak_bytes(self, chunk_elems):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        z = torch.randn(self.B, self.N, 16, device="cuda")
        sim = pairwise_sim(z, "cosine").requires_grad_(True)
        g = torch.Generator(device="cuda").manual_seed(0)
        raw = torch.rand(self.N, self.N, generator=g, device="cuda") < (self.K / self.N / 2)
        pos = (raw | raw.T).expand_as(sim)
        valid = (~torch.eye(self.N, dtype=torch.bool, device="cuda")).expand_as(sim)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        _masked_triplet(sim, pos, valid, 0.5, chunk_elems=chunk_elems).backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    def test_peak_scales_with_chunk_not_p(self):
        """小 chunk 的峰值必须显著低于不分块——只减前向不减反向等于没修。"""
        small = self._peak_bytes(1 << 15)
        big = self._peak_bytes(1 << 30)
        assert small < big / 3, (f"分块峰值 {small/2**20:.0f} MiB 未显著低于"
                                 f"不分块 {big/2**20:.0f} MiB")
