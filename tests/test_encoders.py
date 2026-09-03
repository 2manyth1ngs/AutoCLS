"""固定 encoder 测试（TECHNICAL_DESIGN.md 模块 4）。

语义："固定"= 架构固定，每个 CLS 各自重新预训练一份权重（非权重冻结）。
dilated_cnn：TS2Vec 式 10 层 dilated CNN，hidden=64，输出 320（AutoCL B.3）。
"""
from pathlib import Path

import pytest
import torch

from autocls.encoders.dilated_cnn import DilatedCNNEncoder
from autocls.encoders.registry import build_encoder
from autocls.encoders.transformer import TransformerEncoder

DEFAULT_ENCODER_YAML = Path(__file__).resolve().parents[1] / "configs" / "encoder.yaml"


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


class TestDilatedCNN:
    def test_output_shape(self):
        enc = DilatedCNNEncoder(input_dims=9)
        x = torch.randn(4, 128, 9, generator=_gen())
        z = enc(x)
        assert z.shape == (4, 128, 320)

    def test_variable_length(self):
        # crop 后长度可变，encoder 必须支持任意 T
        enc = DilatedCNNEncoder(input_dims=9)
        for T in (2, 51, 200):
            z = enc(torch.randn(2, T, 9, generator=_gen()))
            assert z.shape == (2, T, 320)

    def test_default_hyperparams_match_autocl(self):
        # AutoCL B.3：10 层 dilated CNN，hidden 64，输出 320
        enc = DilatedCNNEncoder(input_dims=9)
        assert enc.hidden_dims == 64 and enc.output_dims == 320 and enc.depth == 10

    def test_deterministic_forward(self):
        # 增强是 CLS 的搜索字段，encoder 自身不得引入随机掩码
        # （区别于 TS2Vec 原实现 encoder.py 中 training 时的 binomial mask）
        enc = DilatedCNNEncoder(input_dims=9)
        enc.train()
        x = torch.randn(2, 64, 9, generator=_gen())
        assert torch.equal(enc(x), enc(x))

    def test_gradient_flows(self):
        enc = DilatedCNNEncoder(input_dims=9)
        z = enc(torch.randn(2, 32, 9, generator=_gen()))
        z.sum().backward()
        grads = [p.grad for p in enc.parameters()]
        assert all(g is not None for g in grads)
        assert all(torch.isfinite(g).all() for g in grads)

    def test_nan_input_zeroed(self):
        # 对齐 TS2Vec encoder.py：NaN 时间步置零，输出保持有限
        enc = DilatedCNNEncoder(input_dims=9)
        x = torch.randn(2, 32, 9, generator=_gen())
        x[0, 3] = float("nan")
        assert torch.isfinite(enc(x)).all()


class TestTransformerEncoder:
    def test_output_shape(self):
        enc = TransformerEncoder(input_dims=9)
        z = enc(torch.randn(4, 128, 9, generator=_gen()))
        assert z.shape == (4, 128, 320)

    def test_variable_length(self):
        enc = TransformerEncoder(input_dims=9)
        for T in (2, 51, 200):
            assert enc(torch.randn(2, T, 9, generator=_gen())).shape == (2, T, 320)

    def test_deterministic_forward(self):
        # 与 dilated_cnn 同一约束：encoder 自身不得引入随机性（dropout=0）
        enc = TransformerEncoder(input_dims=9)
        enc.train()
        x = torch.randn(2, 64, 9, generator=_gen())
        assert torch.equal(enc(x), enc(x))

    def test_position_matters(self):
        # 有位置编码：同一时间步集合的不同排列应产生不同表示
        enc = TransformerEncoder(input_dims=4)
        x = torch.randn(1, 16, 4, generator=_gen())
        z_fwd = enc(x)
        z_rev = enc(x.flip(dims=[1]))
        assert not torch.allclose(z_fwd, z_rev.flip(dims=[1]), atol=1e-4)

    def test_gradient_flows(self):
        enc = TransformerEncoder(input_dims=9)
        z = enc(torch.randn(2, 32, 9, generator=_gen()))
        z.sum().backward()
        assert all(p.grad is not None for p in enc.parameters() if p.requires_grad)


class TestRegistry:
    def test_default_yaml_builds_dilated_cnn(self):
        # 关键决策 2：encoder 由 configs/encoder.yaml 切换，默认 dilated_cnn
        enc = build_encoder(input_dims=9, config_path=DEFAULT_ENCODER_YAML)
        assert isinstance(enc, DilatedCNNEncoder)
        assert enc.hidden_dims == 64 and enc.output_dims == 320 and enc.depth == 10

    def test_switch_to_transformer(self, tmp_path):
        cfg = tmp_path / "encoder.yaml"
        cfg.write_text(
            "encoder: transformer\ntransformer: {d_model: 32, output_dims: 64, "
            "nhead: 2, num_layers: 1, dim_feedforward: 64}\n",
            encoding="utf-8",
        )
        enc = build_encoder(input_dims=9, config_path=cfg)
        assert isinstance(enc, TransformerEncoder)
        assert enc(torch.randn(2, 16, 9, generator=_gen())).shape == (2, 16, 64)

    def test_unknown_encoder_raises(self, tmp_path):
        cfg = tmp_path / "encoder.yaml"
        cfg.write_text("encoder: mamba\n", encoding="utf-8")
        with pytest.raises((KeyError, ValueError)):
            build_encoder(input_dims=9, config_path=cfg)


# ---------------------------------------------------------------------------
# 阶梯专用开关（全量协议对齐 M7）：TS2Vec 官方 encoder 训练期的 binomial 输入掩码
# （ts2vec_model/encoder.py:53-66，p=0.5）与 repr_dropout 0.1。两者只在 --ladder
# 阶段可用；build_encoder() 默认两开关皆关——造 label 永不触发（默认路径逐位不变）。
# ---------------------------------------------------------------------------


class TestLadderSwitches:
    def _pair(self, **kw):
        torch.manual_seed(11)
        base = DilatedCNNEncoder(input_dims=5, output_dims=16, hidden_dims=8, depth=2)
        torch.manual_seed(11)
        alt = DilatedCNNEncoder(input_dims=5, output_dims=16, hidden_dims=8, depth=2, **kw)
        return base, alt

    def test_defaults_off(self):
        enc = DilatedCNNEncoder(input_dims=9)
        assert enc.input_mask == "none"
        assert isinstance(enc.repr_dropout, torch.nn.Identity)

    def test_build_encoder_default_never_enables(self):
        # 造 label 的唯一入口 build_encoder()：默认必须两开关皆关
        enc = build_encoder(input_dims=9, config_path=DEFAULT_ENCODER_YAML)
        assert enc.input_mask == "none"
        assert isinstance(enc.repr_dropout, torch.nn.Identity)

    def test_build_encoder_overrides(self):
        enc = build_encoder(input_dims=9, config_path=DEFAULT_ENCODER_YAML,
                            overrides={"input_mask": "binomial", "repr_dropout": 0.1})
        assert enc.input_mask == "binomial"
        assert isinstance(enc.repr_dropout, torch.nn.Dropout) and enc.repr_dropout.p == 0.1
        enc2 = build_encoder(input_dims=9, config_path=DEFAULT_ENCODER_YAML, overrides=None)
        assert enc2.input_mask == "none"

    def test_eval_forward_equals_baseline(self):
        base, alt = self._pair(input_mask="binomial", repr_dropout=0.1)
        base.eval()
        alt.eval()
        x = torch.randn(3, 32, 5, generator=_gen(2))
        assert torch.equal(base(x), alt(x))

    def test_train_forward_stochastic_with_binomial(self):
        _, alt = self._pair(input_mask="binomial")
        alt.train()
        x = torch.randn(3, 32, 5, generator=_gen(2))
        assert not torch.equal(alt(x), alt(x))

    def test_train_forward_stochastic_with_repr_dropout(self):
        _, alt = self._pair(repr_dropout=0.1)
        alt.train()
        x = torch.randn(3, 32, 5, generator=_gen(2))
        assert not torch.equal(alt(x), alt(x))

    def test_train_forward_default_deterministic(self):
        base, alt = self._pair()          # 默认：训练态仍确定（既有 test_deterministic_forward 语义）
        alt.train()
        x = torch.randn(3, 32, 5, generator=_gen(2))
        assert torch.equal(alt(x), alt(x))
        assert torch.equal(base(x), alt(x))

    def test_state_dict_keys_unchanged(self):
        base, alt = self._pair(input_mask="binomial", repr_dropout=0.1)
        assert list(base.state_dict()) == list(alt.state_dict())
        alt.load_state_dict(base.state_dict())   # 可互相加载

    def test_invalid_input_mask_raises(self):
        with pytest.raises(ValueError):
            DilatedCNNEncoder(input_dims=9, input_mask="continuous")

    def test_binomial_mask_zeroes_whole_timesteps_after_input_fc(self, monkeypatch):
        # 掩码作用在 input_fc 之后、按 (B,T) 整步置零（官方 x[~mask]=0 语义）
        _, alt = self._pair(input_mask="binomial")
        alt.train()
        seen = {}
        orig = alt.feature_extractor.forward

        def spy(h):
            seen["h"] = h.detach().clone()
            return orig(h)

        monkeypatch.setattr(alt.feature_extractor, "forward", spy)
        alt(torch.randn(4, 64, 5, generator=_gen(1)))
        h = seen["h"]                         # (B, hidden, T)
        zero_t = (h == 0).all(dim=1)          # 整步全零
        frac = zero_t.float().mean().item()
        assert 0.3 < frac < 0.7               # Bernoulli(0.5)
        assert torch.equal(h == 0, zero_t.unsqueeze(1).expand_as(h))
