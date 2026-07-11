"""cl_trainer 测试（TECHNICAL_DESIGN.md 模块 3；铁律 2：max_iters 必须显式传参）。

训练量计量单位为 iters（optimizer.step() 参数更新次数），而非 epoch：
路线 B 的多切片任务大小不同，按 epoch 截断会让各任务的实际参数更新次数不一致，
污染跨任务 early-val label。max_iters 保证任何任务/CLS 的训练量精确相同。
"""
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


def _patch_step_counter(monkeypatch):
    """替换 torch.optim.AdamW 为计数子类，返回已创建实例列表以读取 step_count。"""
    created = []
    real_adamw = torch.optim.AdamW

    class CountingAdamW(real_adamw):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.step_count = 0
            created.append(self)

        def step(self, *args, **kwargs):
            self.step_count += 1
            return super().step(*args, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", CountingAdamW)
    return created


class TestPretrainEncoder:
    def test_max_iters_is_required_keyword(self):
        # 铁律 2：early-val 与完整评估禁止混用 → max_iters 无默认值、必须显式传
        sig = inspect.signature(pretrain_encoder)
        p = sig.parameters["max_iters"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY
        assert p.default is inspect.Parameter.empty
        assert "max_epochs" not in sig.parameters  # epoch 计量已废弃

    def test_exact_step_count_across_pass_boundary(self, make_cls, monkeypatch):
        # n=32, batch=8 → 每遍数据 4 次更新；max_iters=10 跨 pass 边界（4+4+2）
        created = _patch_step_counter(monkeypatch)
        pretrain_encoder(
            _small_encoder(), make_cls(), _toy_data(n=32),
            max_iters=10, batch_size=8, seed=0, device="cpu",
        )
        assert len(created) == 1
        assert created[0].step_count == 10

    def test_skipped_tiny_batch_not_counted(self, make_cls, monkeypatch):
        # n=17, batch=8 → 每遍 batch 为 8/8/1，末尾 1 样本无负例被跳过、不计 step
        # → 每遍只有 2 次更新；max_iters=5 需要第 3 遍数据（2+2+1）
        created = _patch_step_counter(monkeypatch)
        losses = pretrain_encoder(
            _small_encoder(), make_cls(), _toy_data(n=17),
            max_iters=5, batch_size=8, seed=0, device="cpu",
        )
        assert created[0].step_count == 5
        assert len(losses) == 3  # 3 遍数据 → 3 个逐 pass 平均损失

    def test_returns_per_pass_losses(self, make_cls):
        # n=32, batch=8 → 4 步/遍；max_iters=10 → 2 个完整 pass + 1 个截断 pass
        losses = pretrain_encoder(
            _small_encoder(), make_cls(), _toy_data(n=32),
            max_iters=10, batch_size=8, seed=0, device="cpu",
        )
        assert len(losses) == 3
        assert all(np.isfinite(l) for l in losses)

    def test_raises_when_no_batch_can_step(self, make_cls):
        # N=1：所有 batch 都 <2 样本被跳过，step 永不推进 → 必须抛错而非死循环
        with pytest.raises(ValueError):
            pretrain_encoder(
                _small_encoder(), make_cls(), _toy_data(n=1),
                max_iters=4, batch_size=8, seed=0, device="cpu",
            )

    def test_loss_decreases(self, make_cls):
        # n=64, batch=16 → 4 步/遍；32 iters = 原 8 epoch 的等量训练
        losses = pretrain_encoder(
            _small_encoder(), make_cls(), _toy_data(n=64),
            max_iters=32, batch_size=16, seed=0, device="cpu",
        )
        assert losses[-1] < losses[0]

    def test_parameters_updated(self, make_cls):
        enc = _small_encoder()
        before = [p.detach().clone() for p in enc.parameters()]
        pretrain_encoder(enc, make_cls(), _toy_data(), max_iters=4, batch_size=8, device="cpu")
        changed = any(not torch.equal(b, p.detach()) for b, p in zip(before, enc.parameters()))
        assert changed

    def test_deterministic_given_seed(self, make_cls):
        results = []
        for _ in range(2):
            torch.manual_seed(123)
            enc = _small_encoder()
            losses = pretrain_encoder(
                enc, make_cls(), _toy_data(), max_iters=4, batch_size=8, seed=7, device="cpu"
            )
            results.append(losses[0])
        assert results[0] == results[1]
