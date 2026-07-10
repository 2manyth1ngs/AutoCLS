"""cl_trainer 测试（TECHNICAL_DESIGN.md 模块 3；铁律 2：max_epochs 必须显式传参）。"""
import inspect

import numpy as np
import pytest
import torch

from autocls.contrastive.cl_trainer import pretrain_encoder
from autocls.encoders.dilated_cnn import DilatedCNNEncoder


def _toy_data(n=32, T=32, C=3, seed=0):
    """两簇可分的玩具时序：类0为正弦+噪声，类1为噪声。"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, T, dtype=np.float32)
    X = rng.normal(0, 0.3, size=(n, T, C)).astype(np.float32)
    X[: n // 2] += np.sin(t)[None, :, None]
    return X


def _small_encoder():
    return DilatedCNNEncoder(input_dims=3, output_dims=32, hidden_dims=16, depth=3)


class TestPretrainEncoder:
    def test_max_epochs_is_required_keyword(self):
        # 铁律 2：early-val(k=5) 与完整评估禁止混用 → max_epochs 无默认值、必须显式传
        sig = inspect.signature(pretrain_encoder)
        p = sig.parameters["max_epochs"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY
        assert p.default is inspect.Parameter.empty

    def test_returns_per_epoch_losses(self, make_cls):
        losses = pretrain_encoder(
            _small_encoder(), make_cls(), _toy_data(),
            max_epochs=2, batch_size=8, seed=0, device="cpu",
        )
        assert len(losses) == 2
        assert all(np.isfinite(l) for l in losses)

    def test_loss_decreases(self, make_cls):
        losses = pretrain_encoder(
            _small_encoder(), make_cls(), _toy_data(n=64),
            max_epochs=8, batch_size=16, seed=0, device="cpu",
        )
        assert losses[-1] < losses[0]

    def test_parameters_updated(self, make_cls):
        enc = _small_encoder()
        before = [p.detach().clone() for p in enc.parameters()]
        pretrain_encoder(enc, make_cls(), _toy_data(), max_epochs=1, batch_size=8, device="cpu")
        changed = any(not torch.equal(b, p.detach()) for b, p in zip(before, enc.parameters()))
        assert changed

    def test_deterministic_given_seed(self, make_cls):
        results = []
        for _ in range(2):
            torch.manual_seed(123)
            enc = _small_encoder()
            losses = pretrain_encoder(
                enc, make_cls(), _toy_data(), max_epochs=1, batch_size=8, seed=7, device="cpu"
            )
            results.append(losses[0])
        assert results[0] == results[1]
