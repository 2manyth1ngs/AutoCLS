"""WP17-ACTS S3-O（results/acts/s3o/PROTOCOL.md）——先于实现。覆盖：多通道编码函数（逐通道编码后跨通道均值）、第一道门只看 E_T 族、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _g():
    return importlib.import_module("scripts.102_acts_s3o")


class _FakeW:
    def encode(self, data, *, encoding_window=None, batch_size=None):
        assert encoding_window == "full_series" and data.shape[2] == 1
        return np.stack([data[:, :, 0].max(1), data[:, :, 0].min(1)], 1)          # (N·C, 2)


def test_encode_fn_channel_mean():
    g = _g()
    X = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)                      # N=2, T=5, C=3
    R = g.channel_mean_encoder(_FakeW())(X)
    assert R.shape == (2, 2)
    assert np.allclose(R[0], [X[0].max(0).mean(), X[0].min(0).mean()])


def test_gate():
    g = _g()
    assert g.gate1({("et50", 10.0, 1): 0.03, ("pt4+sh4", 100.0, 1): 0.002}) is True
    assert g.gate1({("etF", 10.0, 1): 0.01, ("pt4+sh4", 100.0, 1): 0.05}) is False


def test_cli():
    a = _g().build_parser().parse_args(["--compute-et", "--parts", "12", "--part", "5"])
    assert a.compute_et and a.parts == 12 and a.part == 5
