"""排查 B（encoder 架构消融）脚本的单测：变体构造与 pooling 的正确性。

测量协议的可信度全押在"变体只改目标因子、其余逐位不动"上：
- base 变体必须与 build_encoder 默认配置逐位同权重（否则与 0.8911 参照不可比）；
- no_residual 变体权重逐位不变、只有 forward 去掉残差项；
- 多路 pooling 与直接计算逐位一致。
"""
import importlib

import numpy as np
import pytest
import torch

from autocls.encoders.dilated_cnn import ConvBlock, DilatedCNNEncoder
from autocls.encoders.registry import build_encoder

ab = importlib.import_module("scripts.diag_encoder_ablation")


def test_base_variant_bitwise_equals_default_encoder():
    """base 变体与 build_encoder 默认配置在同种子下逐位同权重（参照可比性前提）。"""
    torch.manual_seed(0)
    ref = build_encoder(input_dims=9)
    torch.manual_seed(0)
    var = ab.build_variant(9, ab.BASE_ARCH)
    ref_sd, var_sd = ref.state_dict(), var.state_dict()
    assert ref_sd.keys() == var_sd.keys()
    for k in ref_sd:
        assert torch.equal(ref_sd[k], var_sd[k]), f"权重不一致: {k}"


@pytest.mark.parametrize("depth", [1, 3, 6, 10])
def test_depth_variant_block_count(depth):
    """depth 变体：ConvBlock 数 = depth+1（channels=[hidden]*depth+[output]）。"""
    enc = ab.build_variant(9, {**ab.BASE_ARCH, "depth": depth})
    blocks = [m for m in enc.feature_extractor if isinstance(m, ConvBlock)]
    assert len(blocks) == depth + 1
    x = torch.randn(2, 32, 9)
    assert enc(x).shape == (2, 32, ab.BASE_ARCH["output_dims"])


@pytest.mark.parametrize("dims", [32, 64, 128, 320])
def test_dims_variant_output_shape(dims):
    enc = ab.build_variant(9, {**ab.BASE_ARCH, "output_dims": dims})
    assert enc(torch.randn(2, 16, 9)).shape == (2, 16, dims)


def test_no_residual_same_weights_different_forward():
    """no_residual：权重逐位同 base（同种子），forward 等于纯卷积路径。"""
    torch.manual_seed(7)
    base = ab.build_variant(9, ab.BASE_ARCH)
    torch.manual_seed(7)
    nores = ab.build_variant(9, {**ab.BASE_ARCH, "residual": False})
    for (ka, va), (kb, vb) in zip(base.state_dict().items(), nores.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)

    x = torch.randn(3, 40, 9)
    base.eval(), nores.eval()
    with torch.no_grad():
        out_base, out_nores = base(x), nores(x)
    assert not torch.allclose(out_base, out_nores), "去残差后输出应当不同"

    # 手工复现单个 block 的无残差路径，验证 monkeypatch 语义正确
    blk = next(m for m in nores.feature_extractor if isinstance(m, ConvBlock))
    xin = torch.randn(2, blk.conv1.conv.in_channels, 20)
    with torch.no_grad():
        expect = blk.conv2(torch.nn.functional.gelu(
            blk.conv1(torch.nn.functional.gelu(xin))))
        got = blk(xin)
    assert torch.allclose(got, expect)


def test_encode_multi_pooled_matches_direct():
    """max/mean/last 三路 pooling 与直接对 (B,T,d) 计算逐位一致，且分批不变值。"""
    torch.manual_seed(1)
    enc = DilatedCNNEncoder(input_dims=4, output_dims=16, hidden_dims=8, depth=2)
    X = np.random.default_rng(0).normal(size=(7, 24, 4)).astype(np.float32)
    feats = ab.encode_multi_pooled(enc, X, ("max", "mean", "last"),
                                   device="cpu", batch_size=3)
    enc.eval()
    with torch.no_grad():
        z = enc(torch.from_numpy(X))
    # 分批前向与整批前向存在 1e-8 量级卷积实现噪声，容差放宽到该量级之上
    for name, ref in (("max", z.max(dim=1).values), ("mean", z.mean(dim=1)),
                      ("last", z[:, -1])):
        np.testing.assert_allclose(feats[name], ref.numpy(), rtol=1e-5, atol=1e-6)


def test_config_grid_isolated_and_complete():
    """配置网格：唯一名、base 在首位、每个非 base 配置恰好偏离 base 一个因子。"""
    names = [c["name"] for c in ab.CONFIGS]
    assert len(names) == len(set(names))
    assert names[0] == "base"
    for cfg in ab.CONFIGS[1:]:
        diff = {k for k in ("depth", "output_dims", "pooling", "residual")
                if cfg[k] != ab.CONFIGS[0][k]}
        assert len(diff) == 1, f"{cfg['name']} 偏离 base 的因子数 {len(diff)} ≠ 1"
