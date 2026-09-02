"""MeanPoolTaskEncoder 单测（实验 1 绕过 SetTF 臂；results/e_degradation_diag/PROTOCOL.md §3）。

任务无关模块（铁律 7）：只吃 (n_windows, T, D) 张量，不 import task_types。
"""
import numpy as np
import pytest
import torch

from autocls.comparator.mean_pool_encoder import MeanPoolTaskEncoder
from autocls.comparator.tclsc_model import TCLSC

D = 6


def _win(n, t, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, t, D, generator=g)


class TestMeanPool:
    def test_set_encode_is_mean_over_windows_and_time(self):
        enc = MeanPoolTaskEncoder(dim_input=D)
        x1, x2 = _win(4, 5, 0), _win(7, 3, 1)  # 窗口数/长度可不同
        out = enc.set_encode([x1, x2])
        assert out.shape == (2, D)
        assert enc.nz == D
        assert torch.allclose(out[0], x1.mean(dim=(0, 1)))
        assert torch.allclose(out[1], x2.mean(dim=(0, 1)))
        assert torch.allclose(enc(x1), out[0])

    def test_no_parameters(self):
        enc = MeanPoolTaskEncoder(dim_input=D)
        assert sum(p.numel() for p in enc.parameters()) == 0

    def test_standardizer_applied(self):
        mu = np.arange(D, dtype=np.float32)
        sd = np.full(D, 2.0, dtype=np.float32)
        enc = MeanPoolTaskEncoder(dim_input=D, mean=mu, std=sd)
        x = _win(3, 4, 2)
        expect = (x.mean(dim=(0, 1)) - torch.from_numpy(mu)) / torch.from_numpy(sd)
        assert torch.allclose(enc.set_encode([x])[0], expect)

    def test_fit_standardizer_uses_only_given_tasks(self):
        windows = {"a": _win(3, 4, 3).numpy(), "b": _win(3, 4, 4).numpy(),
                   "c": _win(3, 4, 5).numpy() * 100}
        mu, sd = MeanPoolTaskEncoder.fit_standardizer(windows, ["a", "b"])
        pooled = np.stack([windows["a"].mean(axis=(0, 1)), windows["b"].mean(axis=(0, 1))])
        assert np.allclose(mu, pooled.mean(axis=0))
        assert np.allclose(sd, pooled.std(axis=0, ddof=0).clip(min=1e-6))
        assert mu.shape == (D,) and sd.shape == (D,)

    def test_fit_standardizer_floors_zero_std(self):
        w = {"a": np.ones((2, 3, D), dtype=np.float32), "b": np.ones((2, 3, D), dtype=np.float32)}
        _mu, sd = MeanPoolTaskEncoder.fit_standardizer(w, ["a", "b"])
        assert (sd > 0).all()

    def test_drop_in_for_tclsc(self):
        # 作为 set_encoder 塞进 T-CLSC：前向可跑、梯度流到 set_fc、E 随任务变
        torch.manual_seed(0)
        model = TCLSC(cls_dim=3, set_encoder=MeanPoolTaskEncoder(dim_input=D),
                      embedding_dim=8, fz=8)
        E = model.encode_tasks([_win(4, 5, 0), _win(4, 5, 1)])
        assert not torch.allclose(E[0], E[1])
        a = torch.rand(2, 3)
        out = model(a, a.flip(0), E)
        out.sum().backward()
        assert model.set_fc[0].weight.grad is not None
        assert out.shape == (2,)
