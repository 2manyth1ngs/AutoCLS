"""对比损失与对比对构造测试（AutoCL 2.2 节；TECHNICAL_DESIGN.md 模块 3）。

最小路径覆盖：cosine 相似度 + InfoNCE + instance 对比。
补全阶段追加：dot/euclidean、Triplet、temporal/cross-scale、层级 pooling。
"""
import math

import pytest
import torch

from autocls.contrastive.losses import contrastive_loss, cross_sim, pairwise_sim
from autocls.contrastive.pair_construction import (
    cross_scale_contrast_loss,
    hierarchical_pool,
    instance_contrast_loss,
    temporal_contrast_loss,
    total_contrastive_loss,
)


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


# ---------- 相似度 ----------

class TestPairwiseSim:
    def test_cosine_known_values(self):
        z = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0]])
        sim = pairwise_sim(z, "cosine")
        assert sim.shape == (3, 3)
        assert torch.allclose(sim.diagonal(), torch.ones(3), atol=1e-6)
        assert abs(sim[0, 1].item()) < 1e-6      # 正交
        assert abs(sim[0, 2].item() + 1.0) < 1e-6  # 反向

    def test_batched_shape(self):
        z = torch.randn(7, 6, 4, generator=_gen())
        assert pairwise_sim(z, "cosine").shape == (7, 6, 6)

    def test_dot_known_values(self):
        z = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
        sim = pairwise_sim(z, "dot")
        assert torch.allclose(sim, torch.tensor([[5.0, 1.0], [1.0, 10.0]]))

    def test_euclidean_known_values(self):
        # AutoCL 用负欧氏距离作相似度
        z = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        sim = pairwise_sim(z, "euclidean")
        assert torch.allclose(sim, torch.tensor([[0.0, -5.0], [-5.0, 0.0]]), atol=1e-6)

    def test_cross_sim_rectangular(self):
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        sim = cross_sim(a, b, "cosine")
        assert sim.shape == (1, 3)
        assert torch.allclose(sim, torch.tensor([[1.0, 0.0, -1.0]]), atol=1e-6)


# ---------- Triplet（AutoCL 引 [17]，公式未给全，按 margin-based hinge 语义补全） ----------

class TestTriplet:
    def test_hand_computed_value(self):
        # anchor0: pos=1 (s=2), neg=2 (s=1)，τ=1，margin=1
        # loss = relu(1 + 1 - 2) = 0；anchor1 对称：pos=0 (s=2)，neg=2 (s=0.5)
        # loss = relu(1 + 0.5 - 2) = 0 → 全部 0
        sim = torch.tensor([[9.0, 2.0, 1.0], [2.0, 9.0, 0.5], [1.0, 0.5, 9.0]])
        pos = torch.zeros(3, 3, dtype=torch.bool)
        pos[0, 1] = pos[1, 0] = True
        loss = contrastive_loss(sim, pos, loss_type="triplet", temperature=1.0)
        assert loss.item() == pytest.approx(0.0)

    def test_hand_computed_nonzero(self):
        # anchor0: relu(1 + 3 - 2) = 2；anchor1: relu(1 + 4 - 2) = 3 → 均值 2.5
        sim = torch.tensor([[9.0, 2.0, 3.0], [2.0, 9.0, 4.0], [3.0, 4.0, 9.0]])
        pos = torch.zeros(3, 3, dtype=torch.bool)
        pos[0, 1] = pos[1, 0] = True
        loss = contrastive_loss(sim, pos, loss_type="triplet", temperature=1.0)
        assert loss.item() == pytest.approx(2.5)

    def test_temperature_scales(self):
        # τ 放大 → 相似度差距被压缩 → hinge 更接近 margin
        sim = torch.tensor([[9.0, 2.0, 3.0], [2.0, 9.0, 4.0], [3.0, 4.0, 9.0]])
        pos = torch.zeros(3, 3, dtype=torch.bool)
        pos[0, 1] = pos[1, 0] = True
        l_small = contrastive_loss(sim, pos, loss_type="triplet", temperature=0.5)
        l_big = contrastive_loss(sim, pos, loss_type="triplet", temperature=100.0)
        assert l_small.item() == pytest.approx(4.0)  # relu(1+6-4)=3, relu(1+8-4)=5 → 4.0
        assert l_big.item() == pytest.approx(1.0, abs=0.05)  # 差距→0，hinge→margin

    def test_no_positive_returns_zero(self):
        sim = torch.eye(3)
        pos = torch.zeros(3, 3, dtype=torch.bool)
        loss = contrastive_loss(sim, pos, loss_type="triplet", temperature=1.0)
        assert loss.item() == 0.0

    def test_gradient_flows(self):
        z = torch.randn(6, 8, generator=_gen(), requires_grad=True)
        sim = pairwise_sim(z, "cosine")
        pos = torch.zeros(6, 6, dtype=torch.bool)
        pos[0, 1] = pos[1, 0] = True
        loss = contrastive_loss(sim, pos, loss_type="triplet", temperature=0.1)
        loss.backward()
        assert z.grad is not None and torch.isfinite(z.grad).all()


# ---------- InfoNCE ----------

class TestInfoNCE:
    def test_hand_computed_value(self):
        # 3 个候选，anchor=0，正样本=1，负样本=2；排除对角线
        sim = torch.tensor([[9.0, 2.0, 1.0], [2.0, 9.0, 1.0], [1.0, 1.0, 9.0]])
        pos = torch.zeros(3, 3, dtype=torch.bool)
        pos[0, 1] = pos[1, 0] = True
        valid = ~torch.eye(3, dtype=torch.bool)
        tau = 0.5
        loss = contrastive_loss(sim, pos, loss_type="info_nce", temperature=tau, valid_mask=valid)
        # anchor0: -log(e^{2/τ}/(e^{2/τ}+e^{1/τ}))；anchor1 对称相同；anchor2 无正样本不计
        expected = -math.log(math.exp(4.0) / (math.exp(4.0) + math.exp(2.0)))
        assert abs(loss.item() - expected) < 1e-5

    def test_temperature_scales_logits(self):
        sim = torch.tensor([[0.0, 1.0, -1.0], [1.0, 0.0, -1.0], [-1.0, -1.0, 0.0]])
        pos = torch.zeros(3, 3, dtype=torch.bool)
        pos[0, 1] = pos[1, 0] = True
        valid = ~torch.eye(3, dtype=torch.bool)
        l_small = contrastive_loss(sim, pos, loss_type="info_nce", temperature=0.1, valid_mask=valid)
        l_big = contrastive_loss(sim, pos, loss_type="info_nce", temperature=100.0, valid_mask=valid)
        # 正样本相似度高于负样本：温度小 → 分布尖锐 → 损失更小
        assert l_small.item() < l_big.item()
        # 温度极大 → 近似均匀分布 → 损失 ≈ log(K)（K=2 个有效候选）
        assert abs(l_big.item() - math.log(2)) < 0.02

    def test_no_positive_rows_ignored(self):
        sim = torch.eye(3) * 5
        pos = torch.zeros(3, 3, dtype=torch.bool)
        valid = ~torch.eye(3, dtype=torch.bool)
        loss = contrastive_loss(sim, pos, loss_type="info_nce", temperature=1.0, valid_mask=valid)
        assert loss.item() == 0.0  # 无 anchor 可用 → 0（与 TS2Vec B=1 行为一致）


# ---------- instance 对比（AutoCL 2.2：跨视图同实例为正，异实例为负） ----------

class TestInstanceContrast:
    def test_aligned_lower_than_shuffled(self):
        g = _gen()
        z1 = torch.randn(8, 16, 32, generator=g)
        z2 = z1 + 0.01 * torch.randn(8, 16, 32, generator=g)
        aligned = instance_contrast_loss(z1, z2, loss_type="info_nce", sim_func="cosine", temperature=1.0)
        shuffled = instance_contrast_loss(z1, z2[torch.randperm(8, generator=g)],
                                          loss_type="info_nce", sim_func="cosine", temperature=1.0)
        assert aligned.item() < shuffled.item()

    def test_single_instance_returns_zero(self):
        # B=1：没有负样本实例，损失为 0（对齐 TS2Vec instance_contrastive_loss）
        z1 = torch.randn(1, 16, 32, generator=_gen())
        z2 = torch.randn(1, 16, 32, generator=_gen(1))
        loss = instance_contrast_loss(z1, z2, loss_type="info_nce", sim_func="cosine", temperature=1.0)
        assert loss.item() == 0.0

    def test_gradient_flows(self):
        z1 = torch.randn(4, 8, 16, generator=_gen(), requires_grad=True)
        z2 = torch.randn(4, 8, 16, generator=_gen(1), requires_grad=True)
        loss = instance_contrast_loss(z1, z2, loss_type="info_nce", sim_func="cosine", temperature=0.1)
        loss.backward()
        assert z1.grad is not None and torch.isfinite(z1.grad).all()

    def test_finite_at_extreme_temperature(self):
        z1 = torch.randn(4, 8, 16, generator=_gen())
        z2 = torch.randn(4, 8, 16, generator=_gen(1))
        for tau in (0.01, 100.0):
            loss = instance_contrast_loss(z1, z2, loss_type="info_nce", sim_func="cosine", temperature=tau)
            assert torch.isfinite(loss)

    def test_all_sim_and_loss_combos_finite(self):
        z1 = torch.randn(4, 8, 16, generator=_gen())
        z2 = torch.randn(4, 8, 16, generator=_gen(1))
        for loss_type in ("info_nce", "triplet"):
            for sim_func in ("dot", "cosine", "euclidean"):
                loss = instance_contrast_loss(
                    z1, z2, loss_type=loss_type, sim_func=sim_func, temperature=1.0
                )
                assert torch.isfinite(loss), (loss_type, sim_func)


# ---------- temporal 对比（AutoCL 2.2：同实例内跨视图同时间步为正） ----------

class TestTemporalContrast:
    def test_aligned_lower_than_time_shuffled(self):
        g = _gen()
        z1 = torch.randn(4, 32, 16, generator=g)
        z2 = z1 + 0.01 * torch.randn(4, 32, 16, generator=g)
        aligned = temporal_contrast_loss(z1, z2, adj_neighbor=False,
                                         loss_type="info_nce", sim_func="cosine", temperature=1.0)
        shuffled = temporal_contrast_loss(z1, z2[:, torch.randperm(32, generator=g)],
                                          adj_neighbor=False,
                                          loss_type="info_nce", sim_func="cosine", temperature=1.0)
        assert aligned.item() < shuffled.item()

    def test_single_timestep_returns_zero(self):
        z1 = torch.randn(4, 1, 16, generator=_gen())
        z2 = torch.randn(4, 1, 16, generator=_gen(1))
        loss = temporal_contrast_loss(z1, z2, adj_neighbor=False,
                                      loss_type="info_nce", sim_func="cosine", temperature=1.0)
        assert loss.item() == 0.0

    def test_adj_neighbor_adds_positives(self):
        # 时间平滑的嵌入：把相邻嵌入纳入正样本应降低损失
        g = _gen()
        base = torch.randn(4, 33, 16, generator=g).cumsum(dim=1) * 0.1  # 平滑轨迹
        z1, z2 = base[:, :32], base[:, 1:]
        l_no = temporal_contrast_loss(z1, z2, adj_neighbor=False,
                                      loss_type="info_nce", sim_func="cosine", temperature=1.0)
        l_adj = temporal_contrast_loss(z1, z2, adj_neighbor=True,
                                       loss_type="info_nce", sim_func="cosine", temperature=1.0)
        assert l_adj.item() != l_no.item()
        assert torch.isfinite(torch.tensor([l_no, l_adj])).all()


# ---------- 层级 pooling（AutoCL 2.2：kernel ∈ {2,3,5} × {avg, max}） ----------

class TestHierarchicalPool:
    def test_avg_kernel2_known_values(self):
        z = torch.tensor([[[1.0], [3.0], [5.0], [7.0]]])  # (1, 4, 1)
        out = hierarchical_pool(z, kernel=2, pool="avg")
        assert torch.allclose(out, torch.tensor([[[2.0], [6.0]]]))

    def test_max_kernel2_known_values(self):
        z = torch.tensor([[[1.0], [3.0], [5.0], [7.0]]])
        out = hierarchical_pool(z, kernel=2, pool="max")
        assert torch.allclose(out, torch.tensor([[[3.0], [7.0]]]))

    def test_ceil_mode_tail_window(self):
        # T=5, k=2 → 3 段（末段单元素）
        z = torch.tensor([[[1.0], [3.0], [5.0], [7.0], [9.0]]])
        out = hierarchical_pool(z, kernel=2, pool="avg")
        assert out.shape == (1, 3, 1)
        assert torch.allclose(out[0, 2], torch.tensor([9.0]))

    def test_kernel5(self):
        z = torch.randn(2, 13, 4, generator=_gen())
        out = hierarchical_pool(z, kernel=5, pool="max")
        assert out.shape == (2, 3, 4)


# ---------- cross-scale 对比（AutoCL 2.2：池化来源位置为正） ----------

class TestCrossScaleContrast:
    def test_pooled_consistent_lower_than_shuffled(self):
        g = _gen()
        z_fine = torch.randn(4, 16, 8, generator=g)
        z_coarse = hierarchical_pool(z_fine, kernel=2, pool="avg")
        consistent = cross_scale_contrast_loss(z_fine, z_coarse, kernel=2,
                                               loss_type="info_nce", sim_func="cosine", temperature=1.0)
        shuffled = cross_scale_contrast_loss(z_fine, z_coarse[:, torch.randperm(8, generator=g)],
                                             kernel=2,
                                             loss_type="info_nce", sim_func="cosine", temperature=1.0)
        assert consistent.item() < shuffled.item()

    def test_single_coarse_position_returns_zero(self):
        # 粗尺度只剩 1 个位置：细尺度候选全为正样本、无负样本 → 0
        z_fine = torch.randn(2, 2, 8, generator=_gen())
        z_coarse = hierarchical_pool(z_fine, kernel=2, pool="avg")
        loss = cross_scale_contrast_loss(z_fine, z_coarse, kernel=2,
                                         loss_type="info_nce", sim_func="cosine", temperature=1.0)
        assert loss.item() == 0.0


# ---------- total_contrastive_loss：按 CLS 组合全部维度 ----------

class TestTotalLoss:
    def test_instance_only_single_scale_matches_direct(self, make_cls):
        z1 = torch.randn(4, 16, 8, generator=_gen())
        z2 = torch.randn(4, 16, 8, generator=_gen(1))
        cfg = make_cls()  # temporal/cross_scale 关，kernel=None
        total = total_contrastive_loss(z1, z2, cfg)
        direct = instance_contrast_loss(z1, z2, loss_type="info_nce",
                                        sim_func="cosine", temperature=0.1)
        assert total.item() == pytest.approx(direct.item())

    def test_full_config_finite_and_backprops(self, make_cls):
        z1 = torch.randn(4, 16, 8, generator=_gen()).requires_grad_(True)
        z2 = torch.randn(4, 16, 8, generator=_gen(1)).requires_grad_(True)
        cfg = make_cls(temporal=True, cross_scale=True, kernel=2, pool="avg",
                       adj_neighbor=True, loss_type="info_nce", sim_func="dot", temperature=1.0)
        loss = total_contrastive_loss(z1, z2, cfg)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(z1.grad).all()

    def test_multi_scale_instance_only(self, make_cls):
        # GGS 形态：仅 instance，但层级 pooling 开启（temporal/cross_scale 有一个开
        # 才有 kernel）—— 这里 temporal=True 触发 kernel=3 的多尺度
        z1 = torch.randn(4, 27, 8, generator=_gen())
        z2 = torch.randn(4, 27, 8, generator=_gen(1))
        cfg = make_cls(temporal=True, kernel=3, pool="max", adj_neighbor=False)
        loss = total_contrastive_loss(z1, z2, cfg)
        assert torch.isfinite(loss)

    def test_triplet_euclidean_combo(self, make_cls):
        z1 = torch.randn(4, 8, 8, generator=_gen())
        z2 = torch.randn(4, 8, 8, generator=_gen(1))
        cfg = make_cls(temporal=True, kernel=2, pool="avg", adj_neighbor=False,
                       loss_type="triplet", sim_func="euclidean", temperature=10.0)
        loss = total_contrastive_loss(z1, z2, cfg)
        assert torch.isfinite(loss)
