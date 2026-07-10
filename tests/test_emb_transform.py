"""嵌入变换测试（AutoCL 2.2"Embedding Transformations"；TECHNICAL_DESIGN.md 模块 3）。

emb_jitter/emb_mask 语义同数据端 jitter/point_mask 但作用于嵌入 (B,T,d)；
norm ∈ {none, layer_norm, l2}。
"""
import torch

from autocls.contrastive.emb_transform import apply_emb_transform


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _z(B=4, T=16, d=32, seed=0):
    return torch.randn(B, T, d, generator=_gen(seed))


class TestEmbTransform:
    def test_identity_config(self, make_cls):
        z = _z()
        cfg = make_cls(emb_jitter=0.0, emb_mask=0.0, norm="none")
        assert torch.equal(apply_emb_transform(z, cfg, generator=_gen()), z)

    def test_emb_jitter_noise_scale(self, make_cls):
        z = torch.zeros(64, 32, 16)
        cfg = make_cls(emb_jitter=0.5)
        out = apply_emb_transform(z, cfg, generator=_gen())
        assert out.shape == z.shape
        assert abs(out.std().item() - 0.5) < 0.02

    def test_emb_mask_ratio(self, make_cls):
        z = torch.ones(64, 64, 8)
        cfg = make_cls(emb_mask=0.4)
        out = apply_emb_transform(z, cfg, generator=_gen())
        # 按时间步掩码：整个时间步置零
        zero_t = (out == 0).all(dim=-1)
        assert abs(zero_t.float().mean().item() - 0.4) < 0.05
        elem_zero = out == 0
        assert torch.equal(elem_zero, zero_t.unsqueeze(-1).expand_as(elem_zero))

    def test_layer_norm(self, make_cls):
        z = _z() * 7 + 3
        cfg = make_cls(norm="layer_norm")
        out = apply_emb_transform(z, cfg, generator=_gen())
        assert torch.allclose(out.mean(dim=-1), torch.zeros_like(out.mean(dim=-1)), atol=1e-5)
        assert torch.allclose(out.std(dim=-1, unbiased=False), torch.ones_like(out.std(dim=-1)), atol=1e-4)

    def test_l2_norm(self, make_cls):
        z = _z() * 5
        cfg = make_cls(norm="l2")
        out = apply_emb_transform(z, cfg, generator=_gen())
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_transform_order_aug_then_norm(self, make_cls):
        # 嵌入增强在前、norm 在后：l2 之后范数仍应为 1（若 norm 在前会被 jitter 破坏）
        z = _z()
        cfg = make_cls(emb_jitter=0.5, norm="l2")
        out = apply_emb_transform(z, cfg, generator=_gen())
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_gradient_flows(self, make_cls):
        z = _z().requires_grad_(True)
        cfg = make_cls(emb_jitter=0.3, emb_mask=0.2, norm="layer_norm")
        out = apply_emb_transform(z, cfg, generator=_gen())
        out.sum().backward()
        assert z.grad is not None and torch.isfinite(z.grad).all()

    def test_boundary_p(self, make_cls):
        z = _z()
        cfg = make_cls(emb_jitter=0.95, emb_mask=0.95, norm="l2")
        out = apply_emb_transform(z, cfg, generator=_gen())
        assert torch.isfinite(out).all()
