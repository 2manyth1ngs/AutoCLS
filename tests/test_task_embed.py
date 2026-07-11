"""task_embed 单测（TECHNICAL_DESIGN.md 模块 5：E = SetTF(TS2Vec(样本序列集))）。

全部用小模型 + 合成数据，CPU 秒级跑完。真实 HAR 上的区分力验证见
scripts/sanity_task_embed.py（sanity check A/B），不放单测。
"""
import inspect

import numpy as np
import pytest
import torch

from autocls.task_embed import embed_task
from autocls.task_embed.set_transformer import TaskSetEncoder
from autocls.task_embed.ts2vec_wrapper import TS2VecWrapper

# 单测用小模型（正式默认 output_dims=256/hidden=64/depth=10，照 AutoCTS++）
TINY = dict(output_dims=16, hidden_dims=8, depth=2)


def _toy(n=12, t=16, c=3, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, t, c)).astype(np.float32)


# ---------------------------------------------------------------- TS2VecWrapper
class TestTS2VecWrapper:
    def test_encode_per_timestep_shape(self):
        X = _toy()
        w = TS2VecWrapper(input_dims=3, device="cpu", seed=0, **TINY)
        w.fit(X, n_iters=2)
        out = w.encode(X)
        assert isinstance(out, np.ndarray)
        assert out.shape == (12, 16, TINY["output_dims"])  # (N, T, D) 逐时间步

    def test_encode_full_series_shape(self):
        X = _toy()
        w = TS2VecWrapper(input_dims=3, device="cpu", seed=0, **TINY)
        w.fit(X, n_iters=2)
        out = w.encode(X, encoding_window="full_series")
        assert out.shape == (12, TINY["output_dims"])  # (N, D) 全序列池化

    def test_fit_counts_param_updates(self):
        # 铁律 2 同款口径：训练量按参数更新次数计，n_iters 必须精确生效
        # （N=12 < batch，1 step/遍 → 5 iters 需跨 5 遍数据）
        X = _toy()
        w = TS2VecWrapper(input_dims=3, device="cpu", seed=0, **TINY)
        w.fit(X, n_iters=5)
        assert w.n_iters == 5

    def test_fit_is_unsupervised(self):
        # 任务嵌入必须是"数据分布特征"而非标签信息（zero-shot 迁移前提）：
        # fit 的签名不得出现标签参数
        params = inspect.signature(TS2VecWrapper.fit).parameters
        assert not {"y", "labels", "targets"} & set(params)

    def test_same_seed_reproducible(self):
        X = _toy()
        outs = []
        for _ in range(2):
            w = TS2VecWrapper(input_dims=3, device="cpu", seed=7, **TINY)
            w.fit(X, n_iters=3)
            outs.append(w.encode(X))
        np.testing.assert_allclose(outs[0], outs[1], atol=1e-6)

    def test_fit_changes_encoding(self):
        X = _toy()
        w = TS2VecWrapper(input_dims=3, device="cpu", seed=0, **TINY)
        before = w.encode(X)
        w.fit(X, n_iters=5)
        after = w.encode(X)
        assert not np.allclose(before, after)


# ---------------------------------------------------------------- TaskSetEncoder
class TestTaskSetEncoder:
    def _enc(self, seed=0):
        torch.manual_seed(seed)
        return TaskSetEncoder(dim_input=16, nz=8)

    def test_single_task_output_shape(self):
        enc = self._enc()
        x = torch.randn(5, 7, 16)  # (n_windows, seq_len, D)
        e = enc(x)
        assert e.shape == (8,)

    def test_set_encode_batch_matches_single(self):
        enc = self._enc()
        x1, x2 = torch.randn(5, 7, 16), torch.randn(4, 7, 16)
        batch = enc.set_encode([x1, x2])
        assert batch.shape == (2, 8)
        torch.testing.assert_close(batch[0], enc(x1), atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(batch[1], enc(x2), atol=1e-5, rtol=1e-4)

    def test_window_permutation_invariant(self):
        # Set-Transformer 语义：任务是样本"集合"，窗口顺序不得影响 E
        enc = self._enc()
        x = torch.randn(6, 7, 16)
        perm = torch.randperm(6)
        torch.testing.assert_close(enc(x), enc(x[perm]), atol=1e-5, rtol=1e-4)

    def test_same_seed_same_weights(self):
        a, b = self._enc(seed=3), self._enc(seed=3)
        for (ka, va), (kb, vb) in zip(
            a.state_dict().items(), b.state_dict().items()
        ):
            assert ka == kb
            torch.testing.assert_close(va, vb, atol=0, rtol=0)


# ---------------------------------------------------------------- embed_task 流水线
PIPE = dict(
    ts2vec_iters=2, sample_num=4, batch_size=8, nz=8, device="cpu", seed=0, **TINY
)


class TestEmbedTask:
    def test_shape_and_reproducible(self):
        X = _toy()
        e1 = embed_task(X, **PIPE)
        e2 = embed_task(X, **PIPE)
        assert e1.shape == (8,)
        np.testing.assert_allclose(e1, e2, atol=1e-6)

    def test_channel_count_agnostic(self):
        # 不同通道数的任务（HAR 9ch vs Epilepsy 1ch vs 通道子集）必须映射到同维 E
        e3 = embed_task(_toy(c=3), **PIPE)
        e5 = embed_task(_toy(c=5), **PIPE)
        assert e3.shape == e5.shape == (8,)

    def test_signature_has_no_labels(self):
        params = inspect.signature(embed_task).parameters
        assert not {"y", "labels", "targets"} & set(params)

    def test_sample_num_larger_than_n_ok(self):
        # 任务样本数 < sample_num 时用全部样本，不得报错
        X = _toy(n=3)
        e = embed_task(X, **{**PIPE, "sample_num": 100})
        assert e.shape == (8,)

    def test_shared_pretrained_ts2vec_skips_refit(self):
        # 传入共享预训练 TS2Vec 时不得重新训练（sanity check 发现每任务独立
        # 重训会让 E 被训练随机性主导，共享冻结编码器是候选修复方案）
        X = _toy()
        w = TS2VecWrapper(input_dims=3, device="cpu", seed=0, **TINY)
        w.fit(X, n_iters=2)
        kwargs = {k: v for k, v in PIPE.items() if k not in ("output_dims", "hidden_dims", "depth")}
        e1 = embed_task(X, ts2vec=w, output_dims=TINY["output_dims"], **kwargs)
        assert w.n_iters == 2  # 未被再次 fit
        e2 = embed_task(X, ts2vec=w, output_dims=TINY["output_dims"], **kwargs)
        assert e1.shape == (8,)
        np.testing.assert_allclose(e1, e2, atol=1e-6)
